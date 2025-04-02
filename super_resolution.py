'''
This script performs the final step after the Slepian fitting where the 
reconstructed VDF is interpolated using thin-plate splines after applying
suprathermal collars to the VDF.
'''

import numpy as np
import emcee, corner
import matplotlib.pyplot as plt; plt.ion()

from line_profiler import profile
from numba import njit

@njit
def Maxwellian_numba(Max_params, xdata, ydata):
    # extracting the fitting parameters
    amp, ux, vxth, vyth = Max_params

    # 100 is scaled with the velocities to make the paramters of order unity
    return (10**amp) * np.exp(-((xdata - 100*ux) / (100*vxth))**2 - (ydata / (100*vyth))**2)

@njit
def biMax_numba(biMax_params, xdata, ydata):
    uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, beam_core_ampratio = biMax_params
    # since amplitudes are in log-scale
    amp_beam = beam_core_ampratio + amp_core

    # converting thermal velocities and anisotropies from log to linear space
    vxth_core = 10**vxth_core
    vthani_core = 10**vthani_core
    vxth_beam = 10**vxth_beam
    vthani_beam = 10**vthani_beam

    # core parameters (only amplitude is still in log scale)
    Max_core = Maxwellian_numba((amp_core, uxcore, vxth_core, vthani_core * vxth_core), xdata, ydata)

    # beam parameters (only amplitude is still in log scale)
    Max_beam = Maxwellian_numba((amp_beam, uxbeam, vxth_beam, vthani_beam * vxth_beam), xdata, ydata)

    # adding the core and beam Maxwellians
    Max_total = (Max_core + Max_beam)

    return Max_total

class supres:
    def __init__(self, vdf, vpara, vperp, baseline=1e-4):
        self.data = np.log10((vdf / np.nanmax(vdf)) + baseline)
        self.xdata = vpara.astype(np.float32)
        self.ydata = vperp.astype(np.float32)
        # to be also used in the forward model when comparing fitted data with input data
        self.baseline = baseline

        # to be filled in at the end of plotting
        self.biMax_fit_params = None

        # creating the dominant weight masks
        self.mask = (self.data <= -0.678) & (self.data >= -2.71)
        # self.mask = (self.data <= -0.678) & (self.data >= -1.737)

        # generating weights once and for all
        self.weight = np.ones_like(self.data)
        self.weight[self.mask] += 100

    def Maxwellian(self, Max_params):
        # extracting the fitting parameters
        amp, ux, vxth, vyth = Max_params

        # 100 is scaled with the velocities to make the paramters of order unity
        return (10**amp) * np.exp(-((self.xdata - 100*ux) / (100*vxth))**2 - (self.ydata / (100*vyth))**2)


    def biMax(self, biMax_params):
        uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, beam_core_ampratio = biMax_params
        # since amplitudes are in log-scale
        amp_beam = beam_core_ampratio + amp_core

        # converting thermal velocities and anisotropies from log to linear space
        vxth_core = 10**vxth_core
        vthani_core = 10**vthani_core
        vxth_beam = 10**vxth_beam
        vthani_beam = 10**vthani_beam

        # core parameters (only amplitude is still in log scale)
        Max_core = self.Maxwellian((amp_core, uxcore, vxth_core, vthani_core * vxth_core))

        # beam parameters (only amplitude is still in log scale)
        Max_beam = self.Maxwellian((amp_beam, uxbeam, vxth_beam, vthani_beam * vxth_beam))

        # adding the core and beam Maxwellians
        Max_total = (Max_core + Max_beam)

        return Max_total

    def biMaxfit(self):
        '''
        Performs an MCMC bi-Maxwellian fitting of the Slepian reconstructed VDF.
        '''
        def log_prior(biMax_params):
            uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, beam_core_ampratio = biMax_params

            if (-10 < uxcore < -1 and -10 < uxbeam < -4 and np.log10(0.2) < vxth_core < np.log10(2) and
                np.log10(1e-2) < vthani_core < np.log10(10) and np.log10(0.2) < vxth_beam < np.log10(2) and
                np.log10(1e-2) < vthani_beam < np.log10(10) and -16 < amp_core < 0 and -16 < beam_core_ampratio < 0):
                return 0.0

            return -np.inf

        def log_probability(biMax_params):
            lp = log_prior(biMax_params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(biMax_params)

        @profile
        def log_likelihood(biMax_params):
            # biMax_model = self.biMax(biMax_params)
            biMax_model = biMax_numba(biMax_params, self.xdata, self.ydata)

            # adding a small baseline; 1e-4 was chosen since the lowest values were around 10^4 less than core
            # this number can be changed later
            logbiMax = np.log10(biMax_model + self.baseline)

            # residual values between data and model fitting
            residual = self.data - logbiMax

            cost = np.dot(self.weight, np.square(residual))

            return -0.5 * cost


        # initializing the biMax_params
        uxcore_init = self.xdata[np.argmax(self.data)] * 1e-2
        uxbeam_init = uxcore_init - 3
        vxth_core_init = np.log10(0.5)
        vthani_core_init = np.log10(1)
        vxth_beam_init = np.log10(0.5)
        vthani_beam_init = np.log10(1)
        amp_core_init = np.max(self.data)
        beam_core_ampratio_init = amp_core_init - 2

        # performing the mcmc of dtw 
        nwalkers = 32
        uxcore_pos = np.random.rand(nwalkers) + uxcore_init
        uxbeam_pos = np.random.rand(nwalkers) + uxbeam_init
        vxth_core_pos = np.random.rand(nwalkers) + vxth_core_init
        vthani_core_pos = np.random.rand(nwalkers) + vthani_core_init
        vxth_beam_pos = np.random.rand(nwalkers) + vxth_beam_init
        vthani_beam_pos = np.random.rand(nwalkers) + vthani_beam_init
        amp_core_pos = (np.random.rand(nwalkers) - 1) + amp_core_init
        beam_core_ampratio_pos = np.random.rand(nwalkers) + beam_core_ampratio_init

        pos = np.array([uxcore_pos, uxbeam_pos, vxth_core_pos, vthani_core_pos,
                        vxth_beam_pos, vthani_beam_pos, amp_core_pos, beam_core_ampratio_pos]).T
        sampler = emcee.EnsembleSampler(nwalkers, 8, log_probability)
        sampler.run_mcmc(pos, 10000, progress=True)

        flat_samples = sampler.get_chain(discard=8000, thin=15, flat=True)

        # converting the temperatures and anisotropies to linear scale
        # flat_samples[:,2:6] = np.power(10, flat_samples[:,2:6])

        labels = ["uxcore", "uxbeam", "vxth_core", "vthani_core", "vxth_beam", "vthani_beam", "amp_core", "beam_core"]
        fig = corner.corner(flat_samples, labels=labels, show_titles=True)

        # find the best estimates and the 1-sigma error bars
        biMax_best_params = np.quantile(flat_samples,q=[0.5],axis=0).squeeze()
        biMax_lower_params = np.quantile(flat_samples,q=[0.14],axis=0).squeeze()
        biMax_upper_params = np.quantile(flat_samples,q=[0.86],axis=0).squeeze()

        self.biMax_fit_params = biMax_best_params

        return biMax_best_params, biMax_lower_params, biMax_upper_params

    def plotfit(self):
        def Maxwellian(Max_params):
            amp, ux, vxth, vyth = Max_params
            return np.power(10., amp) * np.exp(-((xgrid - 100*ux) / (100*vxth))**2 - (ygrid / (100*vyth))**2)

        def biMax(biMax_params):
            uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, beam_core_ampratio = biMax_params
            amp_beam = beam_core_ampratio + amp_core

            # converting from log to linear space
            vxth_core, vthani_core, vxth_beam, vthani_beam = np.power(10, (vxth_core, vthani_core, vxth_beam, vthani_beam))

            # core parameters
            Max_core = Maxwellian((amp_core, uxcore, vxth_core, vthani_core * vxth_core))

            # beam parameters
            Max_beam = Maxwellian((amp_beam, uxbeam, vxth_beam, vthani_beam * vxth_beam))

            # adding the core and beam Maxwellians
            Max_total = Max_core + Max_beam

            return Max_total

        plt.figure()
        # plotting the data
        zeromask = self.data == 0
        plt.tricontourf(self.ydata[~zeromask], self.xdata[~zeromask], self.data[~zeromask],
                        cmap='jet', levels=np.linspace(-4,0,10), alpha=0.7)
        plt.colorbar()
        plt.scatter(self.ydata[~zeromask], self.xdata[~zeromask], c='k', s=2)
        plt.tight_layout()

        # plotting the fitted bi-Maxwellian
        x = np.linspace(np.nanmin(self.xdata), np.nanmax(self.xdata), 120)
        y = np.linspace(-np.nanmax(np.abs(self.ydata)), np.nanmax(np.abs(self.ydata)), 101)
        xgrid, ygrid = np.meshgrid(x, y, indexing='ij')

        biMax_fit_values = np.log10(biMax(self.biMax_fit_params))

        plt.figure()
        plt.contourf(ygrid, xgrid, biMax_fit_values, cmap='jet', levels=np.linspace(-4,0,10), alpha=0.7)
        plt.colorbar()
        plt.scatter(self.ydata[~zeromask], self.xdata[~zeromask], c='k', s=2)
        plt.tight_layout()   

if __name__=='__main__':
    vdf_rec = np.load('vdf_Sleprec.npy').flatten()
    vpara = np.load('vpara.npy').flatten()
    vperp = np.load('vperp.npy').flatten()

    # dummy biMax function compilation
    dummy_params = np.zeros(8)
    _ = biMax_numba(dummy_params, np.zeros_like(vpara), np.zeros_like(vperp))

    # initializing the super-resolution class
    supres_vdf = supres(vdf_rec, vpara, vperp)

    # performing a biMax fitting to obtain the collars
    bimax_best_params, biMax_lower_params, biMax_upper_params = supres_vdf.biMaxfit()

    # plotting the fitted biMax VDF over the Slepian reconstructed data
    supres_vdf.plotfit()
