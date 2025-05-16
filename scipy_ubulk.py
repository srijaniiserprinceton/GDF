import sys
import numpy as np
import astropy.constants as c
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical as c2s
import emcee, corner
import matplotlib.pyplot as plt; plt.ion()
from line_profiler import profile
from scipy.interpolate import BSpline
from scipy.special import eval_legendre
from scipy.optimize import minimize
from datetime import datetime
NAX = np.newaxis

import bsplines
import eval_Slepians
import functions as fn
import coordinate_frame_functions as coor_fn
import misc_plotter
import axisfinder

from scipy.spatial import Delaunay
from tqdm import tqdm

import pickle
import plasmapy.formulary as form
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
mu = 1e-3

def merge_bins(bin_edges, counts, threshold):
    merged_edges = []
    merged_counts = []

    current_count = 0
    start_edge = bin_edges[0]

    for i in range(len(counts)):
        current_count += counts[i]

        # If merged count is at or above threshold, finalize the current bin
        if current_count >= threshold:
            end_edge = bin_edges[i + 1]
            merged_edges.append((start_edge, end_edge))
            merged_counts.append(current_count)
            if i + 1 < len(bin_edges):  # Prepare for next merge
                start_edge = bin_edges[i + 1]
            current_count = 0
        # else continue merging into the next bin

    # Handle any remaining counts (less than threshold at end)
    if current_count > 0:
        if merged_edges:
            # Merge remaining with last bin
            last_start, last_end = merged_edges[-1]
            merged_edges[-1] = (last_start, bin_edges[-1])
            merged_counts[-1] += current_count
        else:
            # If everything was under threshold, merge all into one
            merged_edges.append((bin_edges[0], bin_edges[-1]))
            merged_counts.append(current_count)

    return merged_edges, merged_counts


class gyrovdf:
    def __init__(self, vdf_dict, trange, TH=60, Lmax=12, N2D=None, p=3, spline_mincount=2, count_mask=5, ITERATE=False, CREDENTIALS=None, CLIP=False):
        self.TH = TH  
        self.Lmax = Lmax
        self.N2D = N2D
        self.p = p
        self.count_mask = count_mask 
        self.spline_mincount = spline_mincount
        self.ITERATE = ITERATE

        # loading the Slepians tapers once
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)

        # truncating beyond Shannon number
        if self.N2D is None:
            self.N2D = int(np.sum(self.Slep.V))

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.setup_timestamp_props(vdf_dict, trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)
    
    def setup_timestamp_props(self, vdf_dict, trange, CREDENTIALS=None, CLIP=False):
        time = vdf_dict.time.data
        energy = vdf_dict.energy.data
        theta = vdf_dict.theta.data
        phi = vdf_dict.phi.data
        vdf = vdf_dict.vdf.data
        count = vdf_dict.counts.data

        # masking the zero count bins where we have no constraints
        vdf[count <= self.count_mask] = np.nan
        vdf[vdf == 0] = np.nan
        self.nanmask = np.isfinite(vdf)

        # get and store the min and maxvalues
        self.minval = np.nanmin(psp_vdf.vdf.data, axis=(1,2,3))
        self.maxval = np.nanmax(psp_vdf.vdf.data, axis=(1,2,3))

        m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
        q_p = 1

        self.velocity = np.sqrt(2 * q_p * energy / m_p)

        # Define the Cartesian Coordinates
        self.vx = self.velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        self.vy = self.velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        self.vz = self.velocity * np.sin(np.radians(theta))

        # filemoms = fn.get_psp_span_mom(trange)
        data = fn.init_psp_moms(trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

        # obtaining the mangnetic field and v_bulk measured
        self.b_span = data.MAGF_INST.data
        self.v_span = data.VEL_INST.data

        # Get the angle between b and v.
        self.theta_bv = np.degrees(np.arccos(np.einsum('ij, ij->i', self.v_span, self.b_span)/(np.linalg.norm(self.v_span, axis=1) * np.linalg.norm(self.b_span, axis=1))))

        self.l3_time = data.Epoch.data     # check to make sure the span moments match the l2 data!
        self.l2_time = time

    def get_coors(self, u_bulk, tidx):
        self.vpara, self.vperp1, self.vperp2, self.vperp = None, None, None, None
        
        # Shift into the plasma frame
        self.ux = self.vx[tidx] - u_bulk[0, NAX, NAX, NAX]
        self.uy = self.vy[tidx] - u_bulk[1, NAX, NAX, NAX]
        self.uz = self.vz[tidx] - u_bulk[2, NAX, NAX, NAX]

        # Rotate the plasma frame data into the magnetic field aligned frame.
        vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(self.ux, self.uy, self.uz,
                                                                        *fn.field_aligned_coordinates(self.b_span[tidx])))
        
        self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
        self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

        # # Check angle between flow and magnetic field. 
        if (self.theta_bv[tidx] < 90):
            self.vpara = -1.0 * self.vpara
            self.theta_sign = -1.0
        else: self.theta_sign = 1.0
        # NOTE: NEED TO CHECK ABOVE CALCULATION.
        # self.sign = -1.0*(np.sign(np.median(vpara)))

        # Boosting the vparallel
        self.vshift = np.linalg.norm(self.v_span, axis=1)
        
        self.vpara -= self.vshift[tidx,NAX,NAX,NAX]

        # converting the grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90

    def inversion(self, u_bulk, vdfdata, tidx, SUPER=False, NPTS=100):
            def make_knots(tidx):
                self.knots, self.vpara_nonan = None, None

                # finding the minimum and maximum velocities with counts to find the knot locations
                vmin = np.min(self.velocity[tidx, self.nanmask[tidx]])
                vmax = np.max(self.velocity[tidx, self.nanmask[tidx]])
                dlnv = 0.0348
                
                Nbins = int((np.log10(vmax) - np.log10(vmin)) / dlnv)

                # the knot locations
                self.vpara_nonan = self.r_fa[self.nanmask[tidx]] * np.cos(np.radians(self.theta_fa[self.nanmask[tidx]]))
                self.rfac_nonan = self.r_fa[self.nanmask[tidx]]

                counts, bin_edges = np.histogram(np.log10(self.rfac_nonan), bins=Nbins)

                new_edges, _ = merge_bins(bin_edges, counts, self.spline_mincount)
                log_knots = np.sum(new_edges, axis=1)/2

                # discarding knots at counts less than 10 (always discarding the last knot with low count)
                self.knots = np.power(10, log_knots)

                # arranging the knots in an increasing order
                self.knots = np.sort(self.knots)

                # also making the perp grid for future plotting purposes
                self.vperp_nonan = self.r_fa[self.nanmask[tidx]] * np.sin(np.radians(self.theta_fa[self.nanmask[tidx]]))

            def get_Bsplines_scipy():
                t = np.array([self.knots[0] for i in range(self.p)])
                t = np.append(t, self.knots)
                t = np.append(t, np.array([self.knots[-1] for i in range(self.p)]))
                bsp_basis_coefs = np.identity(len(self.knots) + (self.p-1))
                spl = BSpline(t, bsp_basis_coefs, self.p, extrapolate=True)
                self.B_i_n = spl(self.rfac_nonan).T
                self.B_i_n = np.nan_to_num(spl(self.rfac_nonan).T)

            def get_Slepians_scipy():
                self.S_alpha_n = None

                self.theta_nonan = self.theta_fa[self.nanmask[tidx]]

                L = np.arange(0,self.Lmax+1)
                P_scipy = np.asarray([eval_legendre(ell, np.cos(self.theta_nonan * np.pi / 180)) for ell in L])

                # adding the normalization sqrt((2l+1) / 4pi)
                P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]
                S_n_alpha = P_scipy.T @ np.asarray(self.Slep.C)

                # swapping the axes
                self.S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

                self.S_alpha_n = self.S_alpha_n[:self.N2D,:]

            def get_G_matrix():
                self.G_k_n = None
                self.G_i_alpha_n = None
                # taking the product to make the shape (i x alpha x n)
                self.G_i_alpha_n = self.B_i_n[:,NAX,:] * self.S_alpha_n[NAX,:,:]

                # flattening the k=(i, alpha) dimension to make the shape (k x n)
                npoints = len(self.vpara_nonan)
                self.G_k_n = np.reshape(self.G_i_alpha_n, (-1, npoints))


            def inversion(vdfdata):
                # obtaining the coefficients
                G_g = self.G_k_n @ self.G_k_n.T
                I = np.identity(len(G_g))
                coeffs = np.linalg.inv(G_g + mu * I) @ self.G_k_n @ vdfdata

                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ self.G_k_n

                # finding the zeros which need to be masked to avoid bad cost functions
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, coeffs
            
            # NOTE: Likely to remove...
            def iterative_inversion(vdfdata):
                residual = vdfdata.copy()       # We want to fit the residuals
                coeffs   = np.zeros_like(self.G_i_alpha_n[:,:,0]) # Coefficients for each Slepian function

                vdf_rec = np.zeros_like(vdfdata)
                
                G_0_n = self.G_i_alpha_n[:, 0, :]                      # This is now a matrix
                GGT = G_0_n @ G_0_n.T
                I   = np.identity(len(GGT))
                c   = np.linalg.inv(GGT + mu * I) @ G_0_n @ residual # Same as (GG^T + mu)^{-1} @ G @ f
                coeffs[:,0] = c
                
                residual = residual - c @ G_0_n

                G_i_n = self.G_i_alpha_n[:, 1:, :]
                G_k_n = G_i_n.reshape(-1, self.G_k_n.shape[1])
                GGT = G_k_n @ G_k_n.T
                I   = np.identity(len(GGT))
                c   = np.linalg.inv(GGT + mu * I) @ G_k_n @ residual 
                c   = c.reshape(coeffs[:,1:].shape)
                
                coeffs[:,1:] = c
                
                vdf_rec = coeffs.flatten() @ self.G_k_n
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, coeffs

            
            def define_grids(NPTS):
                self.npts = NPTS

                self.v_para_all = np.concatenate([self.vpara_nonan, self.vpara_nonan])
                self.v_perp_all = np.concatenate([-self.vperp_nonan, self.vperp_nonan])

                points = np.vstack([self.v_para_all, self.v_perp_all]).T   # Shape N x 2

                tri = Delaunay(points)    # Define the Delaunay triangulation

                # Generate the regular grid we are interested in.
                # TODO: Replace with a grid function!
                x = np.linspace(0, 1000, NPTS)
                y = np.linspace(-1000, 1000, NPTS)

                xx, yy = np.meshgrid(x, y, indexing='ij')
                self.grid_points = np.vstack([xx.flatten(), yy.flatten()]).T

                inside = tri.find_simplex(self.grid_points) >= 0    # a Mask for the points inside the domain!
                self.hull_mask = inside

                self.super_rfac  = np.sqrt(self.grid_points[:,0]**2 + self.grid_points[:,1]**2) 
                self.super_theta = np.degrees(np.arctan2(self.grid_points[:,1], self.grid_points[:,0])) # stick to convention

            def super_Slepians_scipy():
                self.super_S_alpha_n = None

                L = np.arange(0,self.Lmax+1)
                P_scipy = np.asarray([eval_legendre(ell, np.cos(self.super_theta * np.pi / 180)) for ell in L])
                # adding the normalization sqrt((2l+1) / 4pi)
                P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]
                S_n_alpha = P_scipy.T @ np.asarray(self.Slep.C)

                # swapping the axes
                self.super_S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

                self.super_S_alpha_n = self.super_S_alpha_n[:self.N2D,:]

            def super_Bsplines_scipy():
                self.super_B_i_n = None

                t = np.array([self.knots[0] for i in range(self.p)])
                t = np.append(t, self.knots)
                t = np.append(t, np.array([self.knots[-1] for i in range(self.p)]))
                bsp_basis_coefs = np.identity(len(self.knots) + (self.p-1))
                spl = BSpline(t, bsp_basis_coefs, self.p, extrapolate=True)
                self.super_B_i_n = spl(self.super_rfac).T

                self.super_B_i_n = np.nan_to_num(spl(self.super_rfac).T)

            def super_G_matrix_scipy():
                self.super_G_k_n = None
                self.super_G_i_alpha_n = None
                # taking the product to make the shape (i x alpha x n)
                self.super_G_i_alpha_n = self.super_B_i_n[:,NAX,:] * self.super_S_alpha_n[NAX,:,:]

                # flattening the k=(i, alpha) dimension to make the shape (k x n)
                npoints = len(self.super_rfac)
                self.super_G_k_n = np.reshape(self.super_G_i_alpha_n, (-1, npoints))
 
            def super_inversion(vdfdata):
                # obtaining the coefficients
                G_g = self.G_k_n @ self.G_k_n.T
                I = np.identity(len(G_g))
                coeffs = np.linalg.inv(G_g + mu * I) @ self.G_k_n @ vdfdata

                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ self.G_k_n
                vdf_super = coeffs.flatten() @ self.super_G_k_n

                # finding the zeros which need to be masked to avoid bad cost functions
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, coeffs, vdf_super

            def super_inversion_revised(tidx, vdfdata):
                NSleps = self.S_alpha_n.shape[0]
                coeffs_ref = np.power(10., np.flip(np.arange(-NSleps+1, 1)))

                # obtaining the coefficients
                G_i_alpha_n = coeffs_ref[NAX,:,NAX] * self.G_i_alpha_n
                G_k_n = np.reshape(G_i_alpha_n, (-1, G_i_alpha_n.shape[-1]))

                G_g = G_k_n @ G_k_n.T
                I = np.identity(len(G_g))

                lambda_arr = np.linspace(-9,-1,100)
                model_misfit = []
                data_misfit = []

                # for lam in lambda_arr:
                # for now lambda is 1e-5 a choice
                coeffs = np.linalg.inv(G_g + 1e-5 * I) @ G_k_n @ vdfdata
                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ G_k_n

                data_misfit.append(np.linalg.norm((vdfdata - vdf_rec)))
                model_misfit.append(np.linalg.norm(coeffs))

                # model_misfit = np.asarray(model_misfit)
                # data_misfit = np.asarray(data_misfit)

                
                # sys.exit()

                super_G_i_alpha_n = coeffs_ref[NAX,:,NAX] * self.super_G_i_alpha_n
                super_G_k_n = np.reshape(super_G_i_alpha_n, (-1, super_G_i_alpha_n.shape[-1]))
                vdf_super = coeffs.flatten() @ super_G_k_n

                # finding the zeros which need to be masked to avoid bad cost functions
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, coeffs, vdf_super

            self.get_coors(u_bulk, tidx)
            make_knots(tidx)
            get_Bsplines_scipy()
            get_Slepians_scipy()
            get_G_matrix()

            if SUPER:
                define_grids(NPTS)
                super_Bsplines_scipy()
                super_Slepians_scipy()
                super_G_matrix_scipy()
                return super_inversion(vdfdata)
            
            if self.ITERATE:
                return iterative_inversion(vdfdata)
            
            return inversion(vdfdata)

def loss_fn_Slepians(p0_2d, points, values, n, origin, u, v, tidx):
    p0 = origin + p0_2d[0]*u + p0_2d[1]*v
    pred, __, __ = gvdf_tstamp.inversion(p0, values, tidx)
    return np.mean((values - pred)**2)

def find_symmetry_point(points, values, n, loss_fn, tidx, origin=None):
    # Get basis u, v orthogonal to n
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(arbitrary, n):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    if(origin is None): origin = np.mean(points, axis=0)  # reasonable guess
    res = minimize(loss_fn, x0=[0.0, 0.0], args=(points, values, n, origin, u, v, tidx), method='Powell')
    best_p0 = origin + res.x[0] * u + res.x[1] * v
    return best_p0, res.fun

def vdf_moments(gvdf, vdf_super, tidx):
    minval = gvdf.minval[tidx]
    maxval = gvdf.maxval[tidx]
    grids = gvdf.grid_points
    vx = grids[:,0].reshape(gvdf.npts, gvdf.npts)
    vy = grids[:,1].reshape(gvdf.npts, gvdf.npts)
    dx = vx[1,0] - vx[0,0]
    dy = vy[0,1] - vy[0,0]

    mask = gvdf.hull_mask
    mask2 = grids[mask,1] >= 0

    f_super = np.power(10, vdf_super) * minval

    f_super[f_super > 5*maxval] = 0.0

    density = 2*np.pi*np.sum(grids[mask,1][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5)
    velocity = (2*np.pi*np.sum(grids[mask,0][mask2] * 1e5 * grids[mask,1][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5))

    vpara = (velocity/density)

    m_p = 1.6726e-24        # g        
    k_b = 1.380649e-16      # erg/K

    T_para = (m_p/k_b)*(2*np.pi*np.sum((grids[mask,0][mask2] * 1e5 - vpara)**2 * grids[mask,1][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5)/density)
    T_perp = (m_p/(2*k_b))*(2*np.pi*np.sum((grids[mask,1][mask2] * 1e5)**2 * grids[mask,1][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5)/density)

    T_comp = T_para, T_perp
    T_trace = (T_para + 2*T_perp)/3

    return(density, vpara/1e5, T_comp, T_trace)
    
def plot_span_vs_rec_scatter(tidx, gvdf, vdf_data, vdf_rec):
    # These are for plotting with the tricontourf routine.
    # getting the plasma frame coordinates
    vpara_pf = gvdf.vpara
    vperp_pf = gvdf.vperp
    vpara_nonan = vpara_pf[tidx, gvdf.nanmask[tidx]]
    vperp_nonan = vperp_pf[tidx, gvdf.nanmask[tidx]]

    vdf_nonan = vdf_data
    vdf_rec_nonan = vdf_rec 
    
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

def plot_span_vs_rec_contour(gvdf, vdf_data, vdf_rec, tidx=None, GRID=False, VA=None, SAVE=False):
    if VA:
        v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])/VA
        v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])/VA
        xlabel = r'$v_{\perp}/v_{A}$'
        ylabel = r'$v_{\parallel}/v_{A}$'
    else:
        v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
        v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])
        xlabel = r'$v_{\perp}$'
        ylabel = r'$v_{\parallel}$'

    # v_para_all -= gvdf.fac.vshift[tidx]

    vdf_nonan = vdf_data
    
    vdf_data_all = np.concatenate([vdf_nonan, vdf_nonan])
    vdf_rec_all  = np.concatenate([vdf_rec, vdf_rec])
    
    lvls = np.linspace(0, int(np.log10(gvdf.maxval[tidx])+1) - int(np.log10(gvdf.minval[tidx]) - 1), 10)

    zeromask = vdf_rec_all == 0
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True, layout='constrained')
    a0 = ax[0].tricontourf(v_perp_all, v_para_all, vdf_data_all, 
                           cmap='plasma', levels=lvls)#, levels=np.linspace(-23, -19, 10))
    ax[0].set_xlabel(xlabel, fontsize=12)
    ax[0].set_ylabel(ylabel, fontsize=12)
    ax[0].set_aspect('equal')
    ax[0].set_title('SPAN VDF')

    a1 = ax[1].tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], vdf_rec_all[~zeromask],
                           cmap='plasma', levels=lvls)#, levels=np.linspace(-23, -19, 10))
    ax[1].set_xlabel(xlabel, fontsize=12)
    ax[1].set_aspect('equal')
    ax[1].set_title('Reconstructed VDF')

    plt.colorbar(a1)

    if GRID:
        [ax[i].scatter(v_perp_all[len(v_para_all)//2:,], v_para_all[len(v_para_all)//2:,], color='k', marker='.', s=3) for i in range(2)]

    if SAVE:
        plt.savefig(f'./Figures/span_rec_contour/tricontour_plot_{tidx}')
        plt.close(fig)

    else: plt.show()

def plot_super_resolution(gvdf, tidx, vdf_super, SAVE=False, VDFUNITS=False, VSHIFT=None, DENSITY=None):
    grids = gvdf_tstamp.grid_points
    mask = gvdf_tstamp.hull_mask

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')


    if VDFUNITS:
        f_super = np.power(10, vdf_super) * gvdf.minval[tidx]
        lvls = np.linspace(int(np.log10(gvdf.minval[tidx]) - 1), int(np.log10(gvdf.maxval[tidx])+1), 10)
        if VSHIFT:
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0] - VSHIFT, np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0] - VSHIFT, np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf_tstamp.vshift[tidx], color='k', marker='.', s=3)
            if DENSITY:
                Bmag = np.linalg.norm(gvdf.b_span[tidx])
                VA = form.speeds.Alfven_speed(Bmag * u.nT, DENSITY * u.cm**(-3), ion='p+').to(u.km/u.s)

                ax.arrow(0, 0, 0, VA.value, fc='k', ec='k')


        else:
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.', s=3)
    else:
        ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], vdf_super[mask], levels=np.linspace(0,4.0,10), cmap='plasma')

        ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.', s=3)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(r'$v_{\perp}$', fontsize=19)
    ax.set_ylabel(r'$v_{\parallel}$', fontsize=19)
    ax.set_title(f'Super Resolution | {str(gvdf.l2_time[tidx])[:19]}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_xlim([-400,400])
    ax.set_aspect('equal')

    if SAVE:
        plt.savefig(f'./Figures/super_res/super_resolved_{tidx}_{gvdf.npts}.pdf')
        plt.close(fig)
    else: plt.show()

def plot_gyrospan(gvdf, tidx, vdfdata, SAVE=False, VDFUNITS=False, VSHIFT=None, DENSITY=None):
    v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
    v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])

    vdf_data_all = np.concatenate([vdfdata, vdfdata])    

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')


    if VDFUNITS:
        f_super = np.power(10, vdf_data_all) * gvdf.minval[tidx]
        lvls = np.linspace(int(np.log10(gvdf.minval[tidx]) - 1), int(np.log10(gvdf.maxval[tidx])+1), 10)
        if VSHIFT:
            ax1 = ax.tricontourf(v_perp_all, v_para_all - VSHIFT, np.log10(f_super), levels=lvls, cmap='plasma')
            if DENSITY:
                Bmag = np.linalg.norm(gvdf.b_span[tidx])
                VA = form.speeds.Alfven_speed(Bmag * u.nT, DENSITY * u.cm**(-3), ion='p+').to(u.km/u.s)

                ax.arrow(0, 0, 0, VA.value, fc='k', ec='k')


        else:
            ax1 = ax.tricontourf(v_perp_all, v_para_all - VSHIFT, np.log10(f_super), levels=lvls, cmap='plasma')
    else:
        ax1 = ax.tricontourf(v_perp_all, v_para_all - VSHIFT, vdf_data_all, levels=np.linspace(0,4.0,10), cmap='plasma')
        
    ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf_tstamp.vshift[tidx], color='k', marker='.', s=3)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(r'$v_{\perp}$', fontsize=19)
    ax.set_ylabel(r'$v_{\parallel}$', fontsize=19)
    ax.set_title(f'Super Resolution | {str(gvdf.l2_time[tidx])[:19]}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim([-400,400])
    ax.set_aspect('equal')

def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

@profile
def main(start_idx = 0, Nsteps = None):
    # the dictionary elements
    v_yz_corr  = {}
    v_yz_lower = {}
    v_yz_upper = {}
    dens = {}
    vels = {}
    v_rec = {}

    # the dictionary that is finally saved as a .pkl file
    vdf_rec_bundle = {}
    gyro_bundle = {}

    if(Nsteps is None): Nsteps = len(psp_vdf.time.data)

    for tidx in tqdm(range(start_idx, start_idx + Nsteps)):
        # Check the l2 and l3 times match (to ensure selection of correct magnetic field)
        if gvdf_tstamp.l2_time[tidx] != gvdf_tstamp.l3_time[tidx]:
            print('Index mismatch. Skipping time.')
            continue

        # initializing the vdf data to optimize (this is the normalized and logarithmic value)
        vdfdata = np.log10(psp_vdf.vdf.data[tidx, gvdf_tstamp.nanmask[tidx]]/gvdf_tstamp.minval[tidx])

        # initializing the Instrument velocity 
        VX      = gvdf_tstamp.v_span[tidx, 0]
        VY_init = gvdf_tstamp.v_span[tidx, 1]
        VZ_init = gvdf_tstamp.v_span[tidx, 2]
        u_origin = np.asarray([VX, VY_init, VZ_init])

        threeD_points = np.vstack([gvdf_tstamp.vx[tidx][gvdf_tstamp.nanmask[tidx]],
                                   gvdf_tstamp.vy[tidx][gvdf_tstamp.nanmask[tidx]],
                                   gvdf_tstamp.vz[tidx][gvdf_tstamp.nanmask[tidx]]]).T

        bvec = gvdf_tstamp.b_span[tidx] / np.linalg.norm(gvdf_tstamp.b_span[tidx])
        u_corr, __ = find_symmetry_point(threeD_points, vdfdata, bvec, loss_fn_Slepians, tidx, origin=u_origin)

        # gvdf_tstamp.get_coors(u_corr, tidx)
        vdf_inv, zeromask, coeffs, vdf_super = gvdf_tstamp.inversion(u_corr, vdfdata, tidx, SUPER=True, NPTS=1001)
        den, vel, Tcomps, Trace = vdf_moments(gvdf_tstamp, vdf_super, tidx)

        # plot_span_vs_rec_contour(gvdf_tstamp, vdfdata, vdf_inv, GRID=True, tidx=tidx, SAVE=False)
        # plot_super_resolution(gvdf_tstamp, tidx, vdf_super, VDFUNITS=True, VSHIFT=vel, SAVE=False)

        dens[tidx] = den
        vels[tidx] = vel
        
        v_span = gvdf_tstamp.v_span[tidx]
        v_mag  = np.linalg.norm(v_span)
        
        # This tells us how far off our v_parallel is from the defined assumed v_parallel
        delta_v = vel - gvdf_tstamp.vshift[tidx]

        # get the assume u_parallel, u_perp1, and u_perp2. from the set 
        u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*u_corr, *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
        u_xnew, u_ynew, u_znew = fn.inverse_rotate_vector_field_aligned(*np.array([u_para - delta_v, u_perp1, u_perp2]), *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))

        u_adj = np.array([u_xnew, u_ynew, u_znew])
        v_rec[tidx] = u_adj

        quick_bundle = {}
        quick_bundle['u_corr'] = u_corr
        quick_bundle['vdf_rec'] = vdf_inv
        quick_bundle['nanmask'] = gvdf_tstamp.nanmask[tidx]

        gyro_bundle[tidx] = quick_bundle

        bundle = {}
        bundle['den'] = den
        bundle['time'] = gvdf_tstamp.l2_time[tidx]
        bundle['component_temp'] = Tcomps
        bundle['scalar_temp'] = Trace
        bundle['u_final'] = u_adj
        bundle['v_yz_corr'] = v_yz_corr
        bundle['v_yz_lower'] = v_yz_lower
        bundle['v_yz_upper'] = v_yz_upper

        vdf_rec_bundle[tidx] = bundle

    ts0 = datetime.strptime(str(gvdf_tstamp.l2_time[start_idx])[0:26], '%Y-%m-%dT%H:%M:%S.%f')
    ts1 = datetime.strptime(str(gvdf_tstamp.l2_time[start_idx + Nsteps])[0:26], '%Y-%m-%dT%H:%M:%S.%f')
    ymd = ts0.strftime('%Y%m%d')
    a_label = ts0.strftime('%H%M%S')
    b_label = ts1.strftime('%H%M%S')
    write_pickle(vdf_rec_bundle, f'./Outputs/scipy_vdf_rec_data_{ymd}_{a_label}_{b_label}')
    write_pickle(gyro_bundle, f'./Outputs/scipy_gyrobundle_{ymd}_{a_label}_{b_label}')

if __name__=='__main__':
    # Initial Parameters
    # trange = ['2020-01-29T00:00:00', '2020-01-29T23:59:59']
    # trange = ['2020-01-26T00:00:00', '2020-01-26T23:59:59']
    # trange = ['2022-02-25T15:00:00', '2022-02-25T16:00:00']
    # trange = ['2024-12-25T09:00:00', '2024-12-25T12:00:00']
    # trange = ['2024-12-24T09:59:59', '2024-12-24T12:00:00']
    trange = ['2020-01-26T07:12:00', '2020-01-26T07:30:59']
    # trange = ['2025-03-21T13:00:00', '2025-03-21T15:00:00']
    # trange = ['2025-03-22T01:00:00', '2025-03-22T03:00:00']
    credentials = fn.load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    # creds = None
    
    # NOTE: Add to separate initialization script in future. 
    TH         = 60
    LMAX       = 12
    N2D        = 3
    P          = 3
    SPLINE_MINCOUNT   = 7
    COUNT_MASK = 1
    ITERATE    = False
    CLIP       = True

    # Load in the VDFs for given timerange
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=CLIP)

    # initializing the inversion class
    gvdf_tstamp = gyrovdf(psp_vdf, trange, Lmax=LMAX, TH=TH, N2D=N2D, 
                          count_mask=COUNT_MASK, spline_mincount=SPLINE_MINCOUNT,
                          ITERATE=ITERATE, CREDENTIALS=creds, CLIP=CLIP)

    # executing the main scripts here
    main(0, len(psp_vdf.time.data) - 1)