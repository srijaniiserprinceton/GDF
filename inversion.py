import sys
import numpy as np
import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt; plt.ion()
from line_profiler import profile
NAX = np.newaxis

import bsplines
import eval_Slepians
import functions as fn
import coordinate_frame_functions as coor_fn

class gyrovdf:
    def __init__(self, vdf_dict, trange, TH=75, Lmax=20, N2D_restrict=True, p=3):
        self.vdf_dict = vdf_dict
        self.trange = trange

        self.TH = TH
        self.Lmax = Lmax
        self.N2D_restrict = N2D_restrict
        self.p = p

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.fac = coor_fn.fa_coordinates()
        self.fac.get_coors(self.vdf_dict, trange, plasma_frame=True)

    @profile
    def setup_new_inversion(self, tidx, knots=None, plot_basis=False, mincount=7):
        self.vpara_nonan, self.theta_nonan = None, None
        self.B_i_n = None
        self.S_alpha_n = None
        self.G_k_n = None

        if(knots is None):
            self.make_knots(tidx, mincount)
        else:
            self.knots = knots

        # we first make the B-splines
        # self.get_Bsplines(plot_basis)
        self.bsp = bsplines.bsplines(self.knots, self.p)
        self.B_i_n = self.bsp.eval_bsp_basis(self.vpara_nonan)
        # we then get the Slepian functions on the required theta grid
        self.get_Slepians(plot_basis, tidx)
        # then we make the G matrix using the basis functions
        self.get_G_matrix()


    def make_knots(self, tidx, mincount):
        # finding the minimum and maximum velocities with counts to find the knot locations
        vmin = np.min(self.fac.velocity[tidx, self.fac.nanmask[tidx]])
        vmax = np.max(self.fac.velocity[tidx, self.fac.nanmask[tidx]])
        dlnv = 0.0348
        Nbins = int((np.log10(vmax) - np.log10(vmin)) / dlnv)

        # the knot locations
        self.vpara_nonan = self.fac.r_fa[tidx, self.fac.nanmask[tidx]] 
        counts, log_knots = np.histogram(np.log10(self.vpara_nonan), bins=Nbins)

        # discarding knots at counts less than 10 (always discarding the last knot with low count)
        log_knots = log_knots[:-1][counts >= mincount]
        self.knots = np.power(10, log_knots)

        # also making the perp grid for future plotting purposes
        self.vperp_nonan = self.fac.vperp[tidx, self.fac.nanmask[tidx]]

    def get_Bsplines(self, plot_basis):
        # loading the bsplines at the r location grid
        self.bsp = bsplines.bsplines(self.knots, self.p)
        self.B_i_n = self.bsp.eval_bsp_basis(self.vpara_nonan)

        # if(plot_basis): self.plot_bsp()


    def get_Slepians(self, plot_basis, tidx):
        # loading the Slepians at the theta location grid
        self.theta_nonan = self.fac.theta_fa[tidx, self.fac.nanmask[tidx]]
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)
        self.Slep.gen_Slep_basis(self.theta_nonan * np.pi / 180)
        S_n_alpha = self.Slep.G * 1.0
        # swapping the axes
        self.S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

        if(plot_basis): self.plot_Slepian_basis()

        # truncating beyond Shannon number
        N2D = int(np.sum(self.Slep.V))
        if(self.N2D_restrict): self.S_alpha_n = self.S_alpha_n[:N2D,:]


    def get_G_matrix(self):
        # taking the product to make the shape (i x alpha x n)
        G_i_alpha_n = self.B_i_n[:,NAX,:] * self.S_alpha_n[NAX,:,:]

        # flattening the k=(i, alpha) dimension to make the shape (k x n)
        npoints = len(self.vpara_nonan)
        self.G_k_n = np.reshape(G_i_alpha_n, (-1, npoints))


    def inversion(self, tidx):
        # getting the vdf data
        vdf_nonan_data = np.log10(self.vdf_dict.vdf.data[tidx, self.fac.nanmask[tidx]])

        # obtaining the coefficients
        G_g = self.G_k_n @ self.G_k_n.T
        I = np.identity(len(G_g))
        coeffs = np.linalg.pinv(G_g + 1e-3 * I) @ self.G_k_n @ vdf_nonan_data

        # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
        vdf_rec = coeffs @ self.G_k_n

        return vdf_rec

    def plot_bsp(self):
        x_min = np.min(self.knots)
        x_max = np.max(self.knots)
        x = np.linspace(x_min, x_max, num=1000)

        bsp_basis = self.bsp.eval_bsp_basis(x)

        # plotting the B-spline components
        for b in bsp_basis:
            plt.plot(x, b)
        plt.title('B-spline basis elements')

    def plot_Slepian_basis(self):
        # generating a regular grid for background plot
        theta_grid = np.linspace(0, np.pi, 180)
        self.Slep.gen_Slep_basis(theta_grid)
        
        fig, ax = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)

        for i in range(9):
            row, col = i//3, i%3
            ax[row,col].plot(self.Slep.th * 180 / np.pi, self.Slep.G[:,i], 'k', lw=0.5)
            ax[row,col].plot(self.theta_nonan, self.S_alpha_n[i], '.r')
            ax[row,col].axvline(int(self.TH), ls='dashed', color='k')
            ax[row,col].set_xlim([0,None])
            ax[row,col].set_title(f'$\lambda$ = {self.Slep.V[i]:.6f}')


if __name__=='__main__':
    # loading VDF and defining timestamp
    # trange = ['2020-01-29T00:00:00', '2020-01-29T00:00:00']
    # psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=None)
    # tidx = 9355
    trange = ['2020-01-26T00:00:00', '2020-01-26T23:00:00']
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=None)
    tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T14:10:42')))

    # initializing the inversion class
    gvdf = gyrovdf(psp_vdf, trange, N2D_restrict=False)
    gvdf.setup_new_inversion(tidx, plot_basis=True)

    # performing the inversion to get the flattened vdf_rec
    vdf_rec_nonan = gvdf.inversion(tidx)

    # making the scatter plot of the gyrotropic VDF
    plt.figure()
    plt.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=-25, vmax=-19)
    plt.scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=-25, vmax=-19)
    plt.title('Reconstructed VDF')
    plt.colorbar()

    filemoms = fn.get_psp_span_mom(trange)
    data = fn.init_psp_moms(filemoms[0])
    density = data.DENS.data
    avg_den = np.convolve(density, np.ones(10)/10, 'same')      # 1-minute average

    va_vec = ((gvdf.fac.b_span * u.nT) / (np.sqrt(c.m_p * c.mu0 * avg_den[:,None] * u.cm**(-3)))).to(u.km/u.s).value
    va_mag = np.linalg.norm(va_vec, axis=1)

    # These are for plotting with the tricontourf routine.
    # getting the plasma frame coordinates
    # gvdf.fac.get_coors(gvdf.vdf_dict, trange, plasma_frame=True)
    vpara_pf = gvdf.fac.vpara
    vperp_pf = gvdf.fac.vperp
    vpara_nonan = vpara_pf[tidx, gvdf.fac.nanmask[tidx]]
    vperp_nonan = vperp_pf[tidx, gvdf.fac.nanmask[tidx]]

    # v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
    # v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])
    v_para_all = np.concatenate([vpara_nonan, vpara_nonan])
    v_perp_all = np.concatenate([-vperp_nonan, vperp_nonan])
    vdf_nonan = gvdf.vdf_dict.vdf.data[tidx, gvdf.fac.nanmask[tidx]]
    vdf_all = np.concatenate([vdf_nonan, vdf_nonan])

    plt.figure()
    plt.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=np.log10(vdf_nonan), vmin=-25, vmax=-19)
    plt.scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=np.log10(vdf_nonan), vmin=-25, vmax=-19)
    plt.title('SPAN VDF')
    plt.colorbar()


    vdf_rec_all = np.concatenate([vdf_rec_nonan, vdf_rec_nonan])
    # masking the zeros
    zeromask = vdf_rec_all == 0

    plt.figure(figsize=(8,4))
    # plt.tricontourf(v_perp_all[~zeromask]/va_mag[tidx], v_para_all[~zeromask]/va_mag[tidx], np.log10(vdf_all)[~zeromask], cmap='cool')
    plt.tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], np.log10(vdf_all)[~zeromask], cmap='cool')
    plt.xlabel(r'$v_{\perp}/v_{a}$')
    plt.ylabel(r'$v_{\parallel}/v_{a}$')
    plt.title('SPAN VDF')

    # v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
    # v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])
    v_para_all = np.concatenate([vpara_nonan, vpara_nonan])
    v_perp_all = np.concatenate([-vperp_nonan, vperp_nonan])
    vdf_nonan = gvdf.vdf_dict.vdf.data[tidx, gvdf.fac.nanmask[tidx]]
    vdf_all = np.concatenate([vdf_nonan, vdf_nonan])

    plt.figure(figsize=(8,4))
    # plt.tricontourf(v_perp_all[~zeromask]/va_mag[tidx], v_para_all[~zeromask]/va_mag[tidx], vdf_rec_all[~zeromask], cmap='cool')
    plt.tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], vdf_rec_all[~zeromask], cmap='cool')
    plt.xlabel(r'$v_{\perp}/v_{a}$')
    plt.ylabel(r'$v_{\parallel}/v_{a}$')
    plt.title('Reconstructed VDF')