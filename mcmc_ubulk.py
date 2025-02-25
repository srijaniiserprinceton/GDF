import sys
import numpy as np
import astropy.constants as c
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical as c2s
import emcee, corner
import matplotlib.pyplot as plt; plt.ion()
from line_profiler import profile
NAX = np.newaxis

import bsplines
import eval_Slepians
import functions as fn
import coordinate_frame_functions as coor_fn

class gyrovdf:
    def __init__(self, vdf_dict, trange, TH=75, Lmax=20, N2D_restrict=True, p=3, mincount=7):
        self.TH = TH
        self.Lmax = Lmax
        self.N2D_restrict = N2D_restrict
        self.p = p
        self.mincount = 7

        # loading the Slepians tapers once
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.setup_timestamp_props(vdf_dict, trange)
    
    def setup_timestamp_props(self, vdf_dict, trange):
        time = vdf_dict.unix_time.data
        energy = vdf_dict.energy.data
        theta = vdf_dict.theta.data
        phi = vdf_dict.phi.data
        vdf = vdf_dict.vdf.data

        # masking the zero count bins where we have no constraints
        vdf[vdf == 0] = np.nan
        self.nanmask = np.isfinite(vdf)

        m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
        q_p = 1

        self.velocity = np.sqrt(2 * q_p * energy / m_p)

        # Define the Cartesian Coordinates
        self.vx = self.velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        self.vy = self.velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        self.vz = self.velocity * np.sin(np.radians(theta))

        filemoms = fn.get_psp_span_mom(trange)
        data = fn.init_psp_moms(filemoms[0])

        # obtaining the mangnetic field and v_bulk measured
        self.b_span = data.MAGF_INST.data
        self.v_span = data.VEL_INST.data

    def get_coors(self, u_bulk, tidx):
        # Shift into the plasma frame
        ux = self.vx[tidx] - u_bulk[0, NAX, NAX, NAX]
        uy = self.vy[tidx] - u_bulk[1, NAX, NAX, NAX]
        uz = self.vz[tidx] - u_bulk[2, NAX, NAX, NAX]

        # Rotate the plasma frame data into the magnetic field aligned frame.
        vpara, vperp1, vperp2 = np.array(fn.rotateVectorIntoFieldAligned(ux, uy, uz,
                                                                         *fn.field_aligned_coordinates(self.b_span[tidx])))
        self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
        self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

        # Boosting the vparallel
        max_r = np.nanmax(self.vperp/np.tan(np.radians(self.TH)) - np.abs(self.vpara))
        self.vpara -= max_r

        # converting the grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90
        self.phi_fa = np.degrees(phi.value)

    def inversion(self, tidx, vdfdata):
            def make_knots(tidx):
                self.knots, self.vpara_nonan, self.vperp_nonan = None, None, None

                # finding the minimum and maximum velocities with counts to find the knot locations
                vmin = np.min(self.velocity[tidx, self.nanmask[tidx]])
                vmax = np.max(self.velocity[tidx, self.nanmask[tidx]])
                dlnv = 0.0348
                Nbins = int((np.log10(vmax) - np.log10(vmin)) / dlnv)

                # the knot locations
                self.vpara_nonan = self.r_fa[self.nanmask[tidx]] 
                counts, log_knots = np.histogram(np.log10(self.vpara_nonan), bins=Nbins)

                # discarding knots at counts less than 10 (always discarding the last knot with low count)
                log_knots = log_knots[:-1][counts >= self.mincount]
                self.knots = np.power(10, log_knots)

                # also making the perp grid for future plotting purposes
                self.vperp_nonan = self.vperp[self.nanmask[tidx]]


            def get_Bsplines():
                self.B_i_n = None
                # loading the bsplines at the r location grid
                bsp = bsplines.bsplines(self.knots, self.p)
                self.B_i_n = bsp.eval_bsp_basis(self.vpara_nonan)


            def get_Slepians():
                self.S_alpha_n = None
                self.theta_nonan = self.theta_fa[self.nanmask[tidx]]
                self.Slep.gen_Slep_basis(self.theta_nonan * np.pi / 180)
                S_n_alpha = self.Slep.G * 1.0
                # swapping the axes
                self.S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

                # truncating beyond Shannon number
                N2D = int(np.sum(self.Slep.V))
                self.S_alpha_n = self.S_alpha_n[:N2D,:]


            def get_G_matrix():
                self.G_k_n = None
                # taking the product to make the shape (i x alpha x n)
                G_i_alpha_n = self.B_i_n[:,NAX,:] * self.S_alpha_n[NAX,:,:]

                # flattening the k=(i, alpha) dimension to make the shape (k x n)
                npoints = len(self.vpara_nonan)
                self.G_k_n = np.reshape(G_i_alpha_n, (-1, npoints))


            def inversion(tidx, vdfdata):
                # obtaining the coefficients
                G_g = self.G_k_n @ self.G_k_n.T
                I = np.identity(len(G_g))
                coeffs = np.linalg.pinv(G_g + 1e-3 * I) @ self.G_k_n @ vdfdata

                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ self.G_k_n

                return vdf_rec


            make_knots(tidx)
            get_Bsplines()
            get_Slepians()
            get_G_matrix()
            return inversion(tidx, vdfdata)


def log_prior(model_params):
    VT, VN = model_params
    if 50 < VT < 150 and 0 < VN < 50:
        return 0.0
    return -np.inf

def log_probability(model_params, VR, vdfdata, tidx):
    lp = log_prior(model_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(model_params, VR, vdfdata, tidx)

def log_likelihood(model_params, VR, vdfdata, tidx):
    VT, VN = model_params
    u_bulk = np.asarray([VR, VT, VN])
    # get new grids and initialize new inversion
    gvdf_tstamp.get_coors(u_bulk, tidx)
    # perform new inversion using the v_span
    vdf_inv = gvdf_tstamp.inversion(tidx, vdfdata)

    cost = np.sum((vdfdata - vdf_inv)**2)
    return -0.5 * cost


if __name__=='__main__':
    trange = ['2020-01-26T00:00:00', '2020-01-26T23:00:00']
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=None)
    tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T14:10:42')))

    # initializing the inversion class
    gvdf_tstamp = gyrovdf(psp_vdf, trange, N2D_restrict=False)

    # initializing the vdf data to optimize
    vdfdata = psp_vdf.vdf.data[tidx, gvdf_tstamp.nanmask[tidx]]

    # initializing the VR
    VR = gvdf_tstamp.v_span[tidx, 0]

    # performing the mcmc of dtw 
    nwalkers = 5
    VT_pos = np.random.rand(nwalkers) + 100
    VN_pos = np.random.rand(nwalkers) + 25
    pos = np.array([VT_pos, VN_pos]).T
    sampler = emcee.EnsembleSampler(nwalkers, 2, log_probability, args=(VR, vdfdata, tidx))
    sampler.run_mcmc(pos, 1000, progress=True)

    # plotting the results of the emcee
    labels = ["VT", "VN"]
    flat_samples = sampler.get_chain(discard=50, thin=15, flat=True)
    fig = corner.corner(flat_samples, labels=labels, show_titles=True)
    plt.savefig('emcee_ubulk.pdf')

    # printing the 0.5 quantile values
    np.quantile(flat_samples,q=[0.5],axis=0).squeeze()