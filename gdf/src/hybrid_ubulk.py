import sys, time
import importlib
import numpy as np
from astropy.coordinates import cartesian_to_spherical as c2s
import emcee, corner
import matplotlib.pyplot as plt; plt.ion()
from line_profiler import profile
from scipy.interpolate import BSpline
from scipy.special import eval_legendre
from scipy.optimize import minimize
from datetime import datetime
from scipy.spatial import Delaunay
from tqdm import tqdm
import pickle
import warnings

from gdf.src import eval_Slepians
from gdf.src import functions as fn
from gdf.src import misc_funcs as misc_fn
from gdf.src import basis_funcs as basis_fn
from gdf.src import plotter

NAX = np.newaxis
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore")

gvdf_tstamp = None
psp_vdf = None

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
    def __init__(self, vdf_dict, trange, TH=60, Lmax=12, N2D=None, p=3, spline_mincount=2, count_mask=5, mu=1e-3, CREDENTIALS=None, CLIP=False):
        self.TH = TH  
        self.Lmax = Lmax
        self.N2D = N2D
        self.p = p
        self.count_mask = count_mask 
        self.spline_mincount = spline_mincount
        self.mu = mu

        # loading the Slepians tapers once
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)
        # generating the Slepian normalizations to be later used for Bspline regularization
        self.Slep.gen_Slep_norms()

        # truncating beyond Shannon number
        if self.N2D is None:
            self.N2D = int(np.sum(self.Slep.V))
        
        self.Slep.norm = self.Slep.norm[:self.N2D]

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.setup_timewindow(vdf_dict, trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)
    
    def setup_timewindow(self, vdf_dict, trange, CREDENTIALS=None, CLIP=False):
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
        self.minval = np.nanmin(vdf_dict.vdf.data, axis=(1,2,3))
        self.maxval = np.nanmax(vdf_dict.vdf.data, axis=(1,2,3))

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

        # Get the b-unit vector
        self.bvec = self.b_span/np.linalg.norm(self.b_span, axis=1)[:,NAX]

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

        # Boosting the vparallel
        self.vshift = np.linalg.norm(self.v_span, axis=1)
        self.vpara -= self.vshift[tidx,NAX,NAX,NAX]

        # converting the grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90

    def inversion(self, u_bulk, vdfdata, tidx, SUPER=False, NPTS=101):
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

                # adding the first and last points as knots
                log_knots = np.append(new_edges[0][0] - dlnv/2., log_knots)
                log_knots = np.append(log_knots, new_edges[-1][-1] + dlnv/2.)

                # discarding knots at counts less than 10 (always discarding the last knot with low count)
                # removing the first and last unconstrained knots
                self.knots = np.power(10, log_knots)

                # arranging the knots in an increasing order
                self.knots = np.sort(self.knots)

                # also making the perp grid for future plotting purposes
                self.vperp_nonan = self.r_fa[self.nanmask[tidx]] * np.sin(np.radians(self.theta_fa[self.nanmask[tidx]]))

            def get_Bsplines():
                self.B_i_n = basis_fn.get_Bsplines_scipy(self.knots, self.p, self.rfac_nonan)
                
            def get_Slepians():
                self.S_alpha_n = basis_fn.get_Slepians_scipy(self.Slep.C, self.theta_fa[self.nanmask[tidx]], 
                                                             self.Lmax, self.N2D)
                
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
                coeffs = np.linalg.inv(G_g + self.mu * I) @ self.G_k_n @ vdfdata

                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ self.G_k_n

                # finding the zeros which need to be masked to avoid bad cost functions
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask
            
            def define_grids(NPTS):
                self.npts = NPTS

                self.v_para_all = np.concatenate([self.vpara_nonan, self.vpara_nonan])
                self.v_perp_all = np.concatenate([-self.vperp_nonan, self.vperp_nonan])

                points = np.vstack([self.v_para_all, self.v_perp_all]).T   # Shape N x 2

                tri = Delaunay(points)    # Define the Delaunay triangulation

                # Generate the regular grid we are interested in.
                # TODO: Replace with a grid function!
                x = np.linspace(0, 2000, NPTS)
                y = np.linspace(-1000, 1000, NPTS)

                xx, yy = np.meshgrid(x, y, indexing='ij')
                self.grid_points = np.vstack([xx.flatten(), yy.flatten()]).T

                inside = tri.find_simplex(self.grid_points) >= 0    # a Mask for the points inside the domain!
                self.hull_mask = inside

                self.super_rfac  = np.sqrt(self.grid_points[:,0]**2 + self.grid_points[:,1]**2) 
                self.super_theta = np.degrees(np.arctan2(self.grid_points[:,1], self.grid_points[:,0])) # stick to convention

            def super_Bsplines():
                self.super_B_i_n = basis_fn.get_Bsplines_scipy(self.knots, self.p, self.super_rfac)

            def super_Slepians():
                self.super_S_alpha_n = basis_fn.get_Slepians_scipy(self.Slep.C, self.super_theta, 
                                                                   self.Lmax, self.N2D)
 
            def super_G_matrix():
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
                
                # Setup the BSpline regularization
                D_i_i = basis_fn.get_Bspline_second_derivative(self.knots, self.p)

                # Augment the D-matrix
                D = np.kron(D_i_i, np.diag(self.Slep.norm))
                # I = np.identity(len(G_g))
                coeffs = np.linalg.inv(G_g + self.mu * D) @ self.G_k_n @ vdfdata

                # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
                vdf_rec = coeffs @ self.G_k_n
                vdf_super = coeffs.flatten() @ self.super_G_k_n

                # finding the zeros which need to be masked to avoid bad cost functions
                zeromask = vdf_rec == 0

                return vdf_rec, zeromask, vdf_super

            self.get_coors(u_bulk, tidx)
            make_knots(tidx)
            get_Bsplines()
            get_Slepians()
            get_G_matrix()

            if SUPER:
                define_grids(NPTS)
                super_Bsplines()
                super_Slepians()
                super_G_matrix()
                return super_inversion(vdfdata)

            return inversion(vdfdata)

# Perpspace Convention
@profile
def log_prior_perpspace(model_params):
    Vperp1, Vperp2 = model_params
    if -100 < Vperp1 < 100 and -100 < Vperp2 < 100:
        return 0.0
    return -np.inf

@profile
def log_probability_perpspace(model_params, vdfdata, tidx, u_init_scipy, u, v):
    lp = log_prior_perpspace(model_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_perpspace(model_params, u_init_scipy, vdfdata, tidx, u, v)

@profile
def log_likelihood_perpspace(p0_2d, origin, vdfdata, tidx, u, v):
    u_bulk = origin + p0_2d[0]*u + p0_2d[1]*v
    
    # perform new inversion using the v_span
    vdf_inv, zeromask = gvdf_tstamp.inversion(u_bulk, vdfdata, tidx)

    cost = np.sum((vdfdata[~zeromask] - vdf_inv[~zeromask])**2)
    return -0.5 * cost

def loss_fn_Slepians(p0_2d, values, origin, u, v, tidx):
    p0 = origin + p0_2d[0]*u + p0_2d[1]*v
    pred, __ = gvdf_tstamp.inversion(p0, values, tidx)
    return np.mean((values - pred)**2)

def find_symmetry_point(points, values, n, loss_fn, tidx, origin=None, MIN_METHOD='L-BFGS-B'):
    # Get basis u, v orthogonal to n
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(arbitrary, n):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    if(origin is None): origin = np.mean(points, axis=0)  # reasonable guess
    res = minimize(loss_fn, x0=[0.0, 0.0], args=(values, origin, u, v, tidx), method=MIN_METHOD)
    best_p0 = origin + res.x[0] * u + res.x[1] * v
    return best_p0, res.fun, u, v

def project_uncertainty(Sigma_ab, u, v, coord='y'):
    coord_idx = {'x': 0, 'y': 1, 'z': 2}[coord]
    J = np.array([u[coord_idx], v[coord_idx]])  # Projection vector for desired component
    sigma2 = J @ Sigma_ab @ J.T
    return np.sqrt(sigma2)

@profile
def main(start_idx = 0, Nsteps = None, NPTS_SUPER=101, MCMC = False, MCMC_WALKERS=8, MCMC_STEPS=2000, MIN_METHOD='L-BFGS-B', SAVE_FIGS=False, SAVE_PKL=True):
    # the dictionary elements
    vperp_12_corr  = {}
    vperp_12_lower = {}
    vperp_12_upper = {}

    # the dictionary that is finally saved as a .pkl file
    vdf_rec_bundle = {}

    if(Nsteps is None): Nsteps = len(psp_vdf.time.data)

    for tidx in tqdm(range(start_idx, start_idx + Nsteps)):
        # Check the l2 and l3 times match (to ensure selection of correct magnetic field)
        if gvdf_tstamp.l2_time[tidx] != gvdf_tstamp.l3_time[tidx]:
            print('Index mismatch. Skipping time.')
            continue

        # initializing the vdf data to optimize (this is the normalized and logarithmic value)
        vdfdata = np.log10(psp_vdf.vdf.data[tidx, gvdf_tstamp.nanmask[tidx]]/gvdf_tstamp.minval[tidx])

        # initializing the Instrument velocity 
        u_origin = gvdf_tstamp.v_span[tidx,:]

        threeD_points = np.vstack([gvdf_tstamp.vx[tidx][gvdf_tstamp.nanmask[tidx]],
                                   gvdf_tstamp.vy[tidx][gvdf_tstamp.nanmask[tidx]],
                                   gvdf_tstamp.vz[tidx][gvdf_tstamp.nanmask[tidx]]]).T

        u_corr, __, u, v = find_symmetry_point(threeD_points, vdfdata, gvdf_tstamp.bvec[tidx], loss_fn_Slepians, tidx, origin=u_origin, MIN_METHOD=MIN_METHOD)

        u_corr_scipy = u_corr * 1.0
        
        # computing super-resolution and moments from scipy correction
        vdf_inv, _, vdf_super = gvdf_tstamp.inversion(u_corr, vdfdata, tidx, SUPER=True, NPTS=NPTS_SUPER)
        den, vel, Tcomps, Trace = fn.vdf_moments(gvdf_tstamp, vdf_super, tidx)

        # This tells us how far off our v_parallel is from the defined assumed v_parallel
        delta_v = vel - gvdf_tstamp.vshift[tidx]

        # get the assume u_parallel, u_perp1, and u_perp2. from the set 
        u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*u_corr, *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
        u_xnew, u_ynew, u_znew = fn.inverse_rotate_vector_field_aligned(*np.array([u_para - delta_v, u_perp1, u_perp2]), *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
        u_adj = np.array([u_xnew, u_ynew, u_znew])

        # if we want to further refine the estimate and obtain error bounds
        if(MCMC):
            Vperp1, Vperp2 = 0.0, 0.0
            
            # performing the mcmc of centroid finder
            nwalkers = MCMC_WALKERS
            Vperp1_pos = np.random.rand(nwalkers) + Vperp1
            Vperp2_pos = np.random.rand(nwalkers) + Vperp2
            pos = np.array([Vperp1_pos, Vperp2_pos]).T

            sampler = emcee.EnsembleSampler(nwalkers, 2, log_probability_perpspace, args=(vdfdata, tidx, u_adj, u, v))
            sampler.run_mcmc(pos, MCMC_STEPS, progress=False)
            
            # plotting the results of the emcee
            labels = [r"$V_{\perp 1}$", r"$V_{\perp 2}$"]
            flat_samples = sampler.get_chain(flat=True)
            
            if SAVE_FIGS:
                fig = corner.corner(flat_samples, labels=labels, show_titles=True)
                plt.savefig(f'./Figures/mcmc_dists/emcee_ubulk_{tidx}.pdf')
                plt.close(fig)

            # computing the quantile values
            vperp_12_corr[tidx] = np.quantile(flat_samples,q=[0.5],axis=0).squeeze()
            vperp_12_lower[tidx] = np.quantile(flat_samples,q=[0.14],axis=0).squeeze()
            vperp_12_upper[tidx] = np.quantile(flat_samples,q=[0.86],axis=0).squeeze()

            u_corr = u_adj + vperp_12_corr[tidx][0] * u + vperp_12_corr[tidx][1] * v

            # making the uncertainties from the covariance matrix
            vperp_12_covmat = np.cov(flat_samples.T)
            sigma_x = project_uncertainty(vperp_12_covmat, u, v, 'x')
            sigma_y = project_uncertainty(vperp_12_covmat, u, v, 'y')
            sigma_z = project_uncertainty(vperp_12_covmat, u, v, 'z')

            # computing super-resolution and moments from MCMC final correction
            vdf_inv, _, vdf_super = gvdf_tstamp.inversion(u_corr, vdfdata, tidx, SUPER=True, NPTS=NPTS_SUPER)
            den, vel, Tcomps, Trace = fn.vdf_moments(gvdf_tstamp, vdf_super, tidx)

            # This tells us how far off our v_parallel is from the defined assumed v_parallel
            delta_v = vel - gvdf_tstamp.vshift[tidx]

            # get the assume u_parallel, u_perp1, and u_perp2. from the set 
            u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*u_corr, *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
            u_xnew, u_ynew, u_znew = fn.inverse_rotate_vector_field_aligned(*np.array([u_para - delta_v, u_perp1, u_perp2]), *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
            u_adj = np.array([u_xnew, u_ynew, u_znew])

        if SAVE_FIGS:
            plotter.plot_span_vs_rec_contour(gvdf_tstamp, vdfdata, vdf_inv, GRID=True, tidx=tidx, SAVE=SAVE_FIGS)
            plotter.plot_super_resolution(gvdf_tstamp, tidx, vdf_super, VDFUNITS=True, VSHIFT=vel, SAVE=SAVE_FIGS)

        bundle = {}
        bundle['den'] = den
        bundle['time'] = gvdf_tstamp.l2_time[tidx]
        bundle['component_temp'] = Tcomps
        bundle['scalar_temp'] = Trace
        bundle['u_final'] = u_adj
        if(MCMC):
            bundle['u_corr']  = u_corr
            bundle['u_corr_scipy'] = u_corr_scipy
            bundle['vperp_12_covmat'] = vperp_12_covmat
            bundle['sigma_x'] = sigma_x
            bundle['sigma_y'] = sigma_y
            bundle['sigma_z'] = sigma_z

        vdf_rec_bundle[tidx] = bundle

    ts0 = datetime.strptime(str(gvdf_tstamp.l2_time[start_idx])[0:26], '%Y-%m-%dT%H:%M:%S.%f')
    ts1 = datetime.strptime(str(gvdf_tstamp.l2_time[start_idx + Nsteps - 1])[0:26], '%Y-%m-%dT%H:%M:%S.%f')
    ymd = ts0.strftime('%Y%m%d')
    a_label = ts0.strftime('%H%M%S')
    b_label = ts1.strftime('%H%M%S')

    if(SAVE_PKL):
        misc_fn.write_pickle(vdf_rec_bundle, f'./Outputs/scipy_vdf_rec_data_{MCMC_WALKERS}_{MCMC_STEPS}_{ymd}_{a_label}_{b_label}')

def run(config):
    global psp_vdf, gvdf_tstamp

    trange = config.TRANGE
    creds = config.CREDS
    creds  = misc_fn.credential_reader(config.CREDS)

    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=config.CLIP)

    gvdf_tstamp = gyrovdf(psp_vdf, trange,
                          Lmax=config.LMAX,
                          TH=config.TH,
                          N2D=config.N2D,
                          count_mask=config.COUNT_MASK,
                          spline_mincount=config.SPLINE_MINCOUNT,
                          mu=config.MU,
                          CREDENTIALS=creds,
                          CLIP=config.CLIP)

    main(config.START_INDEX, Nsteps=config.NSTEPS,
         NPTS_SUPER=config.NPTS_SUPER,
         MCMC=config.MCMC,
         MCMC_WALKERS=config.MCMC_WALKERS,
         MCMC_STEPS=config.MCMC_STEPS,
         MIN_METHOD=config.MIN_METHOD,
         SAVE_FIGS=config.SAVE_FIGS,
         SAVE_PKL =config.SAVE_PKL)

    return gvdf_tstamp

if __name__=='__main__':
    config_file = sys.argv[1]

    config = importlib.import_module(config_file)
    gvdf_tstamp = run(config)