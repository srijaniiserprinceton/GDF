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
NAX = np.newaxis

import bsplines
import eval_Slepians
import src.functions as fn
import coordinate_frame_functions as coor_fn

from scipy.spatial import Delaunay

class gyrovdf:
    def __init__(self, vdf_dict, trange, TH=75, Lmax=16, N2D_restrict=True, p=3, mincount=2, count_mask=1, CREDENTIALS=None, CLIP=False):
        self.TH = TH
        self.Lmax = Lmax
        self.N2D_restrict = N2D_restrict
        self.p = p
        self.count_mask = count_mask 
        self.mincount = mincount

        # loading the Slepians tapers once
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.setup_timestamp_props(vdf_dict, trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)
    
    def setup_timestamp_props(self, vdf_dict, trange, CREDENTIALS=None, CLIP=False):
        time = vdf_dict.unix_time.data
        energy = vdf_dict.energy.data
        theta = vdf_dict.theta.data
        phi = vdf_dict.phi.data
        vdf = vdf_dict.vdf.data
        count = vdf_dict.counts.data

        print("Clip status:", CLIP)

        # masking the zero count bins where we have no constraints
        vdf[count <= self.count_mask] = np.nan
        vdf[vdf == 0] = np.nan
        self.nanmask = np.isfinite(vdf)

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

        print(len(self.b_span), self.vx.shape, (data.MAGF_INST[tidx]))

    def get_coors(self, u_bulk, tidx):
        self.vpara, self.vperp1, self.vperp2, self.vperp = None, None, None, None

        # Shift into the plasma frame
        self.ux = self.vx[tidx] - u_bulk[0, NAX, NAX, NAX]
        self.uy = self.vy[tidx] - u_bulk[1, NAX, NAX, NAX]
        self.uz = self.vz[tidx] - u_bulk[2, NAX, NAX, NAX]

        # Rotate the plasma frame data into the magnetic field aligned frame.
        vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(self.ux, self.uy, self.uz,
                                                                         *fn.field_aligned_coordinates(self.b_span[tidx])))
        
        # upara, uperp1, uperp2 = np.array(fn.rotate_vector_field_aligned(*self.v_span[tidx],
                                                                        #  *fn.field_aligned_coordinates(self.b_span[tidx])))
        self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
        self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

        # Boosting the vparallel
        # max_r = np.nanmax(self.vperp/np.tan(np.radians(self.TH)) - np.abs(self.vpara))
        self.vshift = np.linalg.norm(self.v_span, axis=1)
        
        self.vpara -= self.vshift[tidx,NAX,NAX,NAX]

        # converting the grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90
        # self.phi_fa = np.degrees(phi.value)

    def inversion(self, tidx, vdfdata):
            def make_knots(tidx):
                self.knots, self.vpara_nonan = None, None

                # finding the minimum and maximum velocities with counts to find the knot locations
                vmin = np.min(self.velocity[tidx, self.nanmask[tidx]])
                vmax = np.max(self.velocity[tidx, self.nanmask[tidx]])
                dlnv = 0.0348
                Nbins = int((np.log10(vmax) - np.log10(vmin)) / dlnv)

                # the knot locations
                self.vpara_nonan = self.r_fa[self.nanmask[tidx]] * np.cos(np.radians(self.theta_fa[self.nanmask[tidx]]))
                counts, log_knots = np.histogram(np.log10(self.vpara_nonan), bins=Nbins)

                # discarding knots at counts less than 10 (always discarding the last knot with low count)
                log_knots = log_knots[:-1][counts >= self.mincount]
                self.knots = np.power(10, log_knots)

                # arranging the knots in an increasing order
                self.knots = np.sort(self.knots)

                # # also making the perp grid for future plotting purposes
                self.vperp_nonan = self.r_fa[self.nanmask[tidx]] * np.sin(np.radians(self.theta_fa[self.nanmask[tidx]]))

            def get_Bsplines():
                self.B_i_n = None
                # loading the bsplines at the r location grid
                bsp = bsplines.bsplines(self.knots, self.p)
                self.B_i_n = bsp.eval_bsp_basis(self.vpara_nonan)

            def get_Bsplines_scipy():
                t = np.array([self.knots[0] for i in range(self.p)])
                t = np.append(t, self.knots)
                t = np.append(t, np.array([self.knots[-1] for i in range(self.p)]))
                bsp_basis_coefs = np.identity(len(self.knots) + (self.p-1))
                spl = BSpline(t, bsp_basis_coefs, self.p)
                self.B_i_n = spl(self.vpara_nonan).T

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

                # truncating beyond Shannon number
                N2D = int(np.sum(self.Slep.V))
                self.S_alpha_n = self.S_alpha_n[:N2D,:]            

            def get_G_matrix():
                self.G_k_n = None
                self.G_i_alpha_n = None
                # taking the product to make the shape (i x alpha x n)
                self.G_i_alpha_n = self.B_i_n[:,NAX,:] * self.S_alpha_n[NAX,:,:]

                # flattening the k=(i, alpha) dimension to make the shape (k x n)
                npoints = len(self.vpara_nonan)
                self.G_k_n = np.reshape(self.G_i_alpha_n, (-1, npoints))


            def inversion(tidx, vdfdata):
                # obtaining the coefficients
                G_g = self.G_k_n @ self.G_k_n.T
                I = np.identity(len(G_g))
                coeffs = np.linalg.inv(G_g + 1e-3 * I) @ self.G_k_n @ vdfdata

                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ self.G_k_n

                # finding the zeros which need to be masked to avoid bad cost functions
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, coeffs
            
            def iterative_inversion(tidx, vdfdata):
                residual = vdfdata.copy()       # We want to fit the residuals
                coeffs   = np.zeros_like(self.G_i_alpha_n[:,:,0]) # Coefficients for each Slepian function

                vdf_rec = np.zeros_like(vdfdata)
                for i in range(self.G_i_alpha_n.shape[1]):     # iterate over number of Slepians
                    G_i_n = self.G_i_alpha_n[:, i, :]               # This is now a single vector
                    GGT = G_i_n @ G_i_n.T
                    I   = np.identity(len(GGT))
                    c   =   np.linalg.inv(GGT + 1e-3 * I) @ G_i_n @ residual # Same as (GG^T + mu)^{-1} @ G @ f
                    coeffs[:,i] = c
                    residual = residual - c @ G_i_n

                    vdf_rec += np.dot(c, G_i_n)
                
                # vdf_rec = coeffs @ self.G_k_n
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, coeffs


            make_knots(tidx)
            get_Bsplines()
            get_Slepians()
            get_G_matrix()
            return inversion(tidx, vdfdata)


@profile
def log_prior(model_params):
    VY, VZ = model_params
    if -200 < VY < 200 and -200 < VZ < 200:
        return 0.0
    return -np.inf

@profile
def log_probability(model_params, VX, vdfdata, tidx):
    lp = log_prior(model_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(model_params, VX, vdfdata, tidx)

@profile
def log_likelihood(model_params, VX, vdfdata, tidx):
    VY, VZ = model_params
    u_bulk = np.asarray([VX, VY, VZ])
    # get new grids and initialize new inversion
    gvdf_tstamp.get_coors(u_bulk, tidx)
    # perform new inversion using the v_span
    vdf_inv, zeromask, _ = gvdf_tstamp.inversion(tidx, vdfdata)

    cost = np.sum((vdfdata[~zeromask] - vdf_inv[~zeromask])**2)
    return -0.5 * cost

def super_resolve(gvdf, coeffs, vdf_inv, Npts=100):
    # This function will define a Delaunay triangulation to define the convex hull 
    # that surrounds the defined data values. 
    
    mask = vdf_inv > -22

    # Get all the points. Notice that we are going to mirror the VDF.
    # These represent the points that have local support of the data!
    # v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
    # v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])
    v_para_all = np.concatenate([gvdf.vpara_nonan[mask], gvdf.vpara_nonan[mask]])
    v_perp_all = np.concatenate([-gvdf.vperp_nonan[mask], gvdf.vperp_nonan[mask]])

    points = np.vstack([v_para_all, v_perp_all]).T   # Shape N x 2

    tri = Delaunay(points)    # Define the Delaunay triangulation

    # Generate the regular grid we are interested in.
    # TODO: Replace with a grid function!
    x = np.linspace(0, 655, Npts)
    y = np.linspace(-1000, 1000, Npts)

    xx, yy = np.meshgrid(x, y, indexing='ij')
    grid_points = np.vstack([xx.flatten(), yy.flatten()]).T

    inside = tri.find_simplex(grid_points) >= 0    # a Mask for the points inside the domain!

    # Convert from the Cartesian coordinates to Polar
    th = np.arctan2(grid_points[:,1], grid_points[:,0])

    L = np.arange(0,gvdf.Lmax+1)
    P_scipy = np.asarray([eval_legendre(ell, np.cos(th)) for ell in L])
    # adding the normalization sqrt((2l+1) / 4pi)
    P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]
    S_n_alpha = P_scipy.T @ np.asarray(gvdf.Slep.C)

    # swapping the axes
    S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

    # truncating beyond Shannon number
    N2D = int(np.sum(gvdf.Slep.V))
    S_alpha_n = S_alpha_n[:N2D,:]  

    t = np.array([gvdf.knots[0] for i in range(gvdf.p)])
    t = np.append(t, gvdf.knots)
    t = np.append(t, np.array([gvdf.knots[-1] for i in range(gvdf.p)]))
    bsp_basis_coefs = np.identity(len(gvdf.knots) + (gvdf.p-1))
    spl = BSpline(t, bsp_basis_coefs, gvdf.p)
    vpara_super = grid_points[:,0]
    B_i_n = spl(vpara_super).T

    # n_i, n_alpha, npts
    G_i_alpha_npts = B_i_n[:,NAX,:] * S_alpha_n[NAX,:,:]

    n_pts = grid_points.shape[0]
    G_k_n = np.reshape(G_i_alpha_npts, (-1, n_pts))

    # Super-resolve
    vdf_super = coeffs @ G_k_n

    plt.figure(figsize=(6,6))
    plt.plot(points[:,0], points[:,1], 'o')
    plt.scatter(grid_points[inside,0], grid_points[inside,1], s=1, color='red')
    plt.title('Grid points inside triangulated domain')
    plt.show()

    mask = inside

    return mask, grid_points, vdf_super

def moments(grids, mask, vdf_super, place_holder):
    dx = np.mean(np.diff(grids[:,0].reshape(), axis=1))
    dy = np.mean(np.diff(grids[:,1].reshape(), axis=0))
    mask2 = grids[mask,1] >= 0
    2*np.pi*np.sum(grids[mask,1][mask2]*1e5 * np.power(10, vdf_super[mask][mask2]) * dx*1e5 * dy*1e5)
    (2*np.pi*np.sum(grids[mask,0][mask2] * 1e5 * grids[mask,1][mask2]*1e5 * np.power(10, vdf_super[mask][mask2]) * dx*1e5 * dy*1e5)/100)/1e5


def plot_span_vs_rec_contour(gvdf, vdf_data, vdf_rec, GRID=False, VA=None):
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

    zeromask = vdf_rec_all == 0
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True, layout='constrained')
    a0 = ax[0].tricontourf(v_perp_all, v_para_all, vdf_data_all, 
                           cmap='plasma')#, levels=np.linspace(-23, -19, 10))
    ax[0].set_xlabel(xlabel, fontsize=12)
    ax[0].set_ylabel(ylabel, fontsize=12)
    ax[0].set_aspect('equal')
    ax[0].set_title('SPAN VDF')
    plt.colorbar(a0)

    a1 = ax[1].tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], vdf_rec_all[~zeromask],
                           cmap='plasma')#, levels=np.linspace(-23, -19, 10))
    ax[1].set_xlabel(xlabel, fontsize=12)
    ax[1].set_aspect('equal')
    ax[1].set_title('Reconstructed VDF')

    plt.colorbar(a1)

    if GRID:
        [ax[i].scatter(v_perp_all[len(v_para_all)//2:,], v_para_all[len(v_para_all)//2:,], color='k', marker='.', s=0.8) for i in range(2)]

    plt.show()

if __name__=='__main__':
    # trange = ['2020-01-29T00:00:00', '2020-01-29T23:59:59']
    trange = ['2020-01-26T00:00:00', '2020-01-26T23:59:59']
    # trange = ['2024-12-24T10:00:00', '2024-12-24T12:00:00']
    # credentials = fn.load_config('./config.json')
    # creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    creds = None
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=True)
    tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T14:10:42')))
    # tidx = 0
    v_yz_corr  = {}
    v_yz_lower = {}
    v_yz_upper = {}
    idx = tidx #np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T05:23:54')))
    for tidx in range(idx, idx+1):
        # initializing the inversion class
        gvdf_tstamp = gyrovdf(psp_vdf, trange, Lmax=12, N2D_restrict=False, count_mask=1, mincount=2, CREDENTIALS=creds, CLIP=True)

        # initializing the vdf data to optimize
        vdfdata = np.log10(psp_vdf.vdf.data[tidx, gvdf_tstamp.nanmask[tidx]]/np.nanmin(psp_vdf.vdf.data[tidx, gvdf_tstamp.nanmask[tidx]]))

        # initializing the VR
        VX = gvdf_tstamp.v_span[tidx, 0]
        VY_init= gvdf_tstamp.v_span[tidx, 1]
        VZ_init= gvdf_tstamp.v_span[tidx, 2]

        u_bulk = np.asarray([VX, VY_init, VZ_init])
        gvdf_tstamp.get_coors(u_bulk, tidx)
        vdf_inv, zeromask, coeffs = gvdf_tstamp.inversion(tidx, vdfdata)
        mask, grids, vdf_super = super_resolve(gvdf_tstamp, coeffs, vdf_inv)

        sys.exit()
        # performing the mcmc of dtw 
        nwalkers = 10
        VY_pos = np.random.rand(nwalkers) + VY_init
        VZ_pos = np.random.rand(nwalkers) + VZ_init
        pos = np.array([VY_pos, VZ_pos]).T
        sampler = emcee.EnsembleSampler(nwalkers, 2, log_probability, args=(VX, vdfdata, tidx))
        sampler.run_mcmc(pos, 1000, progress=True)

        # plotting the results of the emcee
        labels = ["VY", "VZ"]
        flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
        fig = corner.corner(flat_samples, labels=labels, show_titles=True)
        # plt.savefig('emcee_ubulk.pdf')

        vdf_inv, zeromask, coeffs = gvdf_tstamp.inversion(tidx, vdfdata)
        plot_span_vs_rec_contour(gvdf_tstamp, vdfdata, vdf_inv, GRID=True)

        # printing the 0.5 quantile values
        v_yz_corr[tidx] = np.quantile(flat_samples,q=[0.5],axis=0).squeeze()
        v_yz_lower[tidx] = np.quantile(flat_samples,q=[0.14],axis=0).squeeze()
        v_yz_upper[tidx] = np.quantile(flat_samples,q=[0.86],axis=0).squeeze()
