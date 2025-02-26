import sys
import json
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class gyrovdf:
    def __init__(self, vdf_dict, trange, TH=75, Lmax=16, N2D_restrict=True, p=3, ITERATE=False, CREDENTIALS=None, CLIP=False):
        self.vdf_dict = vdf_dict
        self.trange = trange

        self.TH = TH
        self.Lmax = Lmax
        self.N2D_restrict = N2D_restrict
        self.p = p

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.fac = coor_fn.fa_coordinates()
        self.fac.get_coors(self.vdf_dict, trange, plasma_frame=True, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

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

    def super_res(self, coeffs, Nth, Nr):
        theta_sup = np.linspace(-np.max(self.theta_nonan), np.max(self.theta_nonan), Nth)
        r_sup = np.linspace(np.min(self.vpara_nonan), np.max(self.vpara_nonan), Nr)

        self.Slep.gen_Slep_basis(theta_sup * np.pi / 180)
        S_n_alpha = self.Slep.G * 1.0
        # swapping the axes
        self.S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

        # if(plot_basis): self.plot_Slepian_basis()

        # truncating beyond Shannon number
        N2D = int(np.sum(self.Slep.V))
        if(self.N2D_restrict): self.S_alpha_n = self.S_alpha_n[:N2D,:]  

        self.B_i_n = self.bsp.eval_bsp_basis(r_sup)

        # taking the product to make the shape (i x alpha x Nr, Nth)
        G_i_alpha_nr_nth = self.B_i_n[:,NAX,:,NAX] * self.S_alpha_n[NAX,:,NAX,:]
        self.G_k_nr_nth = np.reshape(G_i_alpha_nr_nth, (-1, Nr, Nth))
        vdf_sup = coeffs @ np.moveaxis(self.G_k_nr_nth, 0, 1)
        return(r_sup, theta_sup, vdf_sup)
        

    def inversion(self, tidx):
        # getting the vdf data
        vdf_nonan = self.vdf_dict.vdf.data[tidx, self.fac.nanmask[tidx]]
        self.vdf_nonan_data = np.log10(vdf_nonan/np.min(vdf_nonan))

        # obtaining the coefficients
        G_g = self.G_k_n @ self.G_k_n.T
        I = np.identity(len(G_g))
        coeffs = np.linalg.pinv(G_g + 1e-3 * I) @ self.G_k_n @ self.vdf_nonan_data

        # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
        vdf_rec = coeffs @ self.G_k_n

        return vdf_rec, coeffs

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

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def plot_span_vs_rec_scatter(gvdf, vdf_rec):
    # These are for plotting with the tricontourf routine.
    # getting the plasma frame coordinates
    vpara_pf = gvdf.fac.vpara
    vperp_pf = gvdf.fac.vperp
    vpara_nonan = vpara_pf[tidx, gvdf.fac.nanmask[tidx]]
    vperp_nonan = vperp_pf[tidx, gvdf.fac.nanmask[tidx]]

    v_para_all = np.concatenate([vpara_nonan, vpara_nonan])
    v_perp_all = np.concatenate([-vperp_nonan, vperp_nonan])
    vdf_nonan = gvdf.vdf_nonan_data 
    
    vdf_all = np.concatenate([vdf_nonan, vdf_nonan])
    vdf_rec_all = np.concatenate([vdf_rec, vdf_rec])

    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True, layout='constrained')
    ax0 = ax[0].scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=(vdf_nonan), vmin=0, vmax=4)
    ax[0].scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=(vdf_nonan), vmin=0, vmax=4)
    ax[0].set_title('SPAN VDF')
    ax[0].set_ylabel(r'$v_{\parallel}$')
    ax[0].set_xlabel(r'$v_{\perp}$')
    ax[0].set_aspect('equal')
    # making the scatter plot of the gyrotropic VDF
    ax1 = ax[1].scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=0, vmax=4)
    ax[1].scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=0, vmax=4)
    ax[1].set_title('Reconstructed VDF')
    ax[1].set_xlabel(r'$v_{\perp}$')
    ax[1].set_aspect('equal')
    plt.colorbar(ax1)


    plt.show()

def plot_span_vs_rec_contour(gvdf, vdf_rec, GRID=False):
    # These are for plotting with the tricontourf routine.
    # getting the plasma frame coordinates
    vpara_pf = gvdf.fac.vpara
    vperp_pf = gvdf.fac.vperp
    vpara_nonan = vpara_pf[tidx, gvdf.fac.nanmask[tidx]]
    vperp_nonan = vperp_pf[tidx, gvdf.fac.nanmask[tidx]]

    v_para_all = np.concatenate([vpara_nonan, vpara_nonan])
    v_perp_all = np.concatenate([-vperp_nonan, vperp_nonan])
    vdf_nonan = gvdf.vdf_nonan_data 
    
    vdf_all = np.concatenate([vdf_nonan, vdf_nonan])
    vdf_rec_all = np.concatenate([vdf_rec, vdf_rec])

    zeromask = vdf_rec_all == 0
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True, layout='constrained')
    ax[0].tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], (vdf_all)[~zeromask], cmap='inferno', vmin=0, vmax=4)
    ax[0].set_xlabel(r'$v_{\perp}$')
    ax[0].set_ylabel(r'$v_{\parallel}')
    ax[0].set_aspect('equal')
    ax[0].set_title('SPAN VDF')

    ax[1].tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], vdf_rec_all[~zeromask], cmap='inferno', vmin=0, vmax=4)
    ax[1].set_xlabel(r'$v_{\perp}$')
    ax[1].set_aspect('equal')
    ax[1].set_title('Reconstructed VDF')

    if GRID:
        [ax[i].scatter(vperp_nonan, vpara_nonan, color='k', marker='.', s=0.8) for i in range(2)]

    plt.show()

def plot_super_res(gvdf):
    rsup, thetasup, vdf_rec_sup = gvdf.super_res(coeffs, 180, 200)

    Rsup, Tsup = np.meshgrid(rsup, thetasup, indexing='ij')

    v_para_s = Rsup
    v_perp_s = Rsup * np.tan(np.radians(Tsup))

    plt.figure(figsize=(8,4))
    plt.contourf(v_perp_s, v_para_s, vdf_rec_sup, cmap='plasma', vmin=0, vmax=4, levels=100)
    plt.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.')
    plt.scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.')
    plt.xlabel(r'$v_{\perp}/v_{a}$')
    plt.ylabel(r'$v_{\parallel}/v_{a}$')
    plt.gca().set_aspect('equal')
    plt.title('Reconstructed VDF')

if __name__=='__main__':
    # loading VDF and defining timestamp
    trange = ['2020-01-26T00:00:00', '2020-01-26T23:59:59']
    credentials = load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    # creds = None
    
    # Initialzise the PSP vdf
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=True)

    # Choose a user defined time index
    # tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T00:06:00')))
    tidx = 1136

    # initializing the inversion class
    gvdf = gyrovdf(psp_vdf, trange, N2D_restrict=False, CREDENTIALS=creds, CLIP=True)
    
    # Loop over the specified time indicies.
    gvdf.setup_new_inversion(tidx, plot_basis=False)

    # performing the inversion to get the flattened vdf_rec
    vdf_rec_nonan, coeffs = gvdf.inversion(tidx)

    plot_span_vs_rec_scatter(gvdf, vdf_rec_nonan)
    plot_span_vs_rec_contour(gvdf, vdf_rec_nonan)
    plot_super_res(gvdf)

