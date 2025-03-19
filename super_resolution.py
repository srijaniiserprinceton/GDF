'''
This script performs the final step after the Slepian fitting where the 
reconstructed VDF is interpolated using thin-plate splines after applying
suprathermal collars to the VDF.
'''

import numpy as np
import emcee, corner
import matplotlib.pyplot as plt; plt.ion()

class supres:
    def __init__(self, logvdf, vpara, vperp, maxamp):
        self.data = logvdf
        self.xdata = vpara
        self.ydata = vperp
        self.maxamp = maxamp

        # to be filled in at the end of plotting
        self.biMax_fit_params = None

    def Maxwellian(self, Max_params):
        amp, ux, vxth, vyth = Max_params

        return np.power(10., amp) * np.exp(-((self.xdata - 100*ux) / (100*vxth))**2 - (self.ydata / (100*vyth))**2)

    def biMax(self, biMax_params):
        uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, amp_beam = biMax_params

        # core parameters
        
        Max_core = self.Maxwellian((amp_core, uxcore, vxth_core, vthani_core * vxth_core))

        # beam parameters
        Max_beam = self.Maxwellian((amp_beam, uxbeam, vxth_beam, vthani_beam * vxth_beam))

        # adding the core and beam Maxwellians
        Max_total = (Max_core + Max_beam)

        return Max_total

    def biMaxfit(self):
        '''
        Performs an MCMC bi-Maxwellian fitting of the Slepian reconstructed VDF.
        '''
        def log_prior(biMax_params):
            uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, amp_beam = biMax_params

            if (-10 < uxcore < -1 and -10 < uxbeam < -4 and 0 < vxth_core < 2 and
                1e-2 < vthani_core < 10 and 0 < vxth_beam < 2 and 0.5 < vthani_beam < 2 and
                -16 < amp_core < 16 and -16 < amp_beam < 16):
                return 0.0

            return -np.inf

        def log_probability(biMax_params):
            lp = log_prior(biMax_params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(biMax_params)

        def log_likelihood(biMax_params):
            biMax_model = self.biMax(biMax_params)
            # logbiMax = np.log10(biMax_model) + np.log10(self.maxamp)
            logbiMax = biMax_model * self.maxamp

            cost = np.nansum(np.power(10.,self.data)*(self.data - logbiMax)**2)
            return -0.5 * cost


        # initializing the biMax_params
        uxcore_init = self.xdata[np.argmax(self.data)] * 1e-2  # -5
        uxbeam_init = uxcore_init - 2
        vxth_core_init = 0.5
        vthani_core_init = 1
        vxth_beam_init = 0.5
        vthani_beam_init = 1
        amp_core_init = 0
        amp_beam_init = -1

        # performing the mcmc of dtw 
        nwalkers = 19
        uxcore_pos = np.random.rand(nwalkers) + uxcore_init
        uxbeam_pos = np.random.rand(nwalkers) + uxbeam_init
        vxth_core_pos = np.random.rand(nwalkers) + vxth_core_init
        vthani_core_pos = np.random.rand(nwalkers) + vthani_core_init
        vxth_beam_pos = np.random.rand(nwalkers) + vxth_beam_init
        vthani_beam_pos = np.random.rand(nwalkers) + vthani_beam_init
        amp_core_pos = np.random.rand(nwalkers) + amp_core_init
        amp_beam_pos = np.random.rand(nwalkers) + amp_beam_init

        pos = np.array([uxcore_pos, uxbeam_pos, vxth_core_pos, vthani_core_pos,
                        vxth_beam_pos, vthani_beam_pos, amp_core_pos, amp_beam_pos]).T
        sampler = emcee.EnsembleSampler(nwalkers, 8, log_probability)
        sampler.run_mcmc(pos, 10000, progress=True)

        flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)

        labels = ["uxcore", "uxbeam", "vxth_core", "vthani_core", "vxth_beam", "vthani_beam", "amp_core", "amp_beam"]
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
            uxcore, uxbeam, vxth_core, vthani_core, vxth_beam, vthani_beam, amp_core, amp_beam = biMax_params

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
        plt.tricontourf(self.ydata[~zeromask], self.xdata[~zeromask], np.log10(self.data[~zeromask]),
                        cmap='jet', vmin=0, vmax=4)
        plt.colorbar()
        # plt.contourf(self.xdata, self.ydata, self.data, levels=(0, self.maxamp, 12))

        # plotting the contour of the fit over the data
        x = np.linspace(np.nanmin(self.xdata), np.nanmax(self.xdata), 120)
        y = np.linspace(-np.nanmax(np.abs(self.ydata)), np.nanmax(np.abs(self.ydata)), 101)
        xgrid, ygrid = np.meshgrid(x, y, indexing='ij')

        biMax_fit_values = np.log10(biMax(self.biMax_fit_params)) + np.log10(self.maxamp)
        plt.contour(ygrid, xgrid, biMax_fit_values, cmap='jet', vmin=0, vmax=4)
        # plt.colorbar()
        plt.tight_layout()

        plt.figure()
        plt.contourf(ygrid, xgrid, biMax_fit_values, cmap='jet', vmin=0, vmax=4)
        plt.colorbar()
        plt.tight_layout()

    '''
    def makeCollars(self, biMax_params, valfven, vpara, vperp):


    def get_superres_points(self, vpara, vperp):
    '''

def synthetic_test():
    # initializing the biMax_params
    uxcore = -3
    uxbeam = uxcore -1.7
    vxth_core = 0.5
    vthani_core = 1.0
    vxth_beam = 0.5
    vthani_beam = 1.0
    amp_core = 0
    amp_beam = -2

    biMax_params = (uxcore, uxbeam, vxth_core, vthani_core,
                    vxth_beam, vthani_beam, amp_core, amp_beam)

    # generating a grid
    x, y = np.linspace(-1000, 0, 120), np.linspace(-400, 400, 101)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # generating the data
    genvdf = supres(None, xx.flatten(), yy.flatten(), 1)
    data = genvdf.biMax(biMax_params)

    data = data / np.nanmin(data)

    data_reshaped = np.reshape(data, (len(x), len(y)))

    plt.figure()
    plt.contourf(yy, xx, np.log10(data_reshaped), cmap='jet', levels=12)
    plt.colorbar()
    plt.tight_layout()

    return data, xx.flatten(), yy.flatten()
    

if __name__=='__main__':
    
    vdf_rec = np.load('vdf_Sleprec.npy').flatten()
    vdf_rec = vdf_rec / np.nanmin(vdf_rec) + 1.
    vpara = np.load('vpara.npy').flatten()
    vperp = np.load('vperp.npy').flatten()
    '''
    vdf_rec, vpara, vperp = synthetic_test()
    '''

    supres_vdf = supres(vdf_rec, vpara, vperp, np.nanmax(vdf_rec))
    # supres_vdf = supres(np.log10(vdf_rec), vpara, vperp, np.nanmax(vdf_rec))

    # performing a biMax fitting
    bimax_best_params, biMax_lower_params, biMax_upper_params = supres_vdf.biMaxfit()

    # plotting the fitted biMax VDF over the Slepian reconstructed data
    supres_vdf.plotfit()
