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
from gdf.src import plotter
from gdf.src import polar_cap_inversion as polcap
from gdf.src import cartesian_inversion as cartesian
from gdf.src import hybrid_inversion as hybrid

NAX = np.newaxis
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore")

gvdf_tstamp = None
psp_vdf = None

class gyrovdf:
    def __init__(self, vdf_dict, config, CREDENTIALS=None, CLIP=True):
        """
        Initialization of the gyrovdf dictionary (referred to as gvdf_tstamp in the code).
        
        Parameters
        ----------
        vdf_dict : dicitonary
            Dictionary containing the VDF information created using init_psp_vdf() in functions.py

        config : dictionary
            Dictionary containing the input parameters from the initialization file (e.g., init_gdf.py)

        CREDENTIALS : list (optional)
            List containing (FOR NOW) the SWEAP username and password, for proprietary data loading.
        
        CLIP : bool (optional)
            Flag to perform optional non-clipping of the day's data to the specified trange.
        """
        # the method of inversion
        self.method = config['global']['METHOD']
        # the time window
        self.trange = config['global']['TRANGE']
        # the minimum counts required to be considered in the fitting process
        self.count_threshold = config['global']['COUNT_THRESHOLD']

        #---------------------needed for any inversion method in the gyrocentroid finder-------------------#
        # initializing the polar cap Slepian parameters
        self.init_polcap_params(config['polcap'])

        # the B-spline regularization array which is used only for polar cap and hybrid
        self.mu_arr = np.logspace(-10, -5, 20)

        # initializing cartesian slepian parameters (needed for any method which is not 'polcap')
        if(self.method == 'polcap'):
            self.inversion_code = polcap
            self.plotter_func = plotter.polcap_plotter
        elif(self.method == 'cartesian'):
            self.inversion_code = cartesian
            self.init_cartslep_params(config['cartesian'])
            # initializing the CartSlep class once; like we do for the polcap Slepians
            self.CartSlep = eval_Slepians.Slep_2D_Cartesian()
            self.plotter_func = plotter.cartesian_plotter

        elif(self.method == 'hybrid'):
            self.inversion_code = hybrid
            self.lam = config['hybrid']['LAMBDA']
            self.init_cartslep_params(config['cartesian'])
            # initializing the CartSlep class once; like we do for the polcap Slepians
            self.CartSlep = eval_Slepians.Slep_2D_Cartesian()
            self.plotter_func = plotter.hybrid_plotter
        else:
            print('INVALID CHOICE OF METHOD. CHOOSE BETWEEN polcap, cartesian, hybrid')

        # loading/creating the Slepians tapers once before all inversions start
        # NOTE: this does not evaluate the actual Slepian functions, just creates the coefficients for SH
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)

        # truncating beyond N2D; if None is passed it uses the default Shannon number definition
        if self.N2D_polcap is None:
            self.N2D_polcap = int(np.sum(self.Slep.V))

        # generating the Slepian normalizations to be later used for Bspline regularization (in the D matrix)
        self.Slep.gen_Slep_norms()
        self.Slep.norm = self.Slep.norm[:self.N2D_polcap]

        # obtaining the grid points from PSP VDF (instrument frame)
        self.setup_timewindow(vdf_dict, self.trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

        # this is updated everytime ---> contains the log10 values of the VDF after purging 
        # any of the grids which are below the count threshold.
        self.vdfdata = None
    
    def init_polcap_params(self, polcap_params):
        """
        Initializes the parameters of the polar cap Slepians and stores them as
        attributes of the class.
        """
        self.Lmax = polcap_params['LMAX']
        self.N2D_polcap = polcap_params['N2D_POLCAP']
        self.p = polcap_params['P']
        self.spline_mincount = polcap_params['SPLINE_MINCOUNT']

        # setting the get_coor_supres function depending on if TH is None or specified to a desired value
        if(polcap_params['TH'] is None):
            self.TH = 60   # the default extend of the instrument in instrument field aligned geometry
            self.get_coors_supres = self.get_coors_update_TH
        else:
            self.TH = polcap_params['TH']
            self.get_coors_supres = self.get_coors

        # this is for when we do the polar cap inversion and super-resolution
        self.G_k_n = None
        self.G_i_alpha_n = None
        self.super_G_k_n = None
        self.super_G_i_alpha_n = None
        self.B_i_n = None
        self.S_alpha_n = None
    
    def init_cartslep_params(self, cartslep_params):
        """
        Initializes the parameters of the Cartesian Slepians and stores them as
        attributes of the class.
        """
        # storing the user defined N2D_cart if it is not None
        self.N2D_cart = None
        self.N2D_cart_max = cartslep_params['N2D_CART_MAX']
        self.N2D_cart_default = cartslep_params['N2D_CART']
    
    def setup_timewindow(self, vdf_dict, trange, CREDENTIALS=None, CLIP=False):
        """
        Setting up the L3 data (moments and associated magnetic fields), clipping to the 
        desired time range and filtering the vdf to keep grids above the count threshold.

        Parameters
        ----------
        vdf_dict : dictionary

        trange : string tuple
            The time range provided in the initialization file.

        CREDENTIALS : list (optional)
            The credentials list containing the username and password for SWEAP to access proprietary data.

        CLIP : bool (optional)
            Boolean flag to indicate if we want to clip the data to the desired time window.
        """
        # extracting out all the required variables from the vdf dictionary
        time = vdf_dict.time.data
        energy = vdf_dict.energy.data * 1.0
        theta = vdf_dict.theta.data * 1.0
        phi = vdf_dict.phi.data * 1.0
        vdf = vdf_dict.vdf.data * 1.0
        count = vdf_dict.counts.data * 1.0

        # masking the bins where count is less than COUNT_THRESHOLD
        vdf[count <= self.count_threshold] = np.nan
        vdf[vdf == 0] = np.nan
        self.nanmask = np.isfinite(vdf)

        # store the min and maxvalues rquired for plotting
        self.minval = np.nanmin(vdf, axis=(1,2,3))
        self.maxval = np.nanmax(vdf, axis=(1,2,3))

        m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
        q_p = 1

        # constructing the velocity grid from the energy grid
        self.velocity = np.sqrt(2 * q_p * energy / m_p)

        # Define the Cartesian Coordinates
        self.vx = self.velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        self.vy = self.velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        self.vz = self.velocity * np.sin(np.radians(theta))

        # obtaining the L3 data which contains the magnetic field and partial moments
        data = fn.init_psp_moms(trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

        # obtaining the magnetic field and v_bulk from the SWEAP L3 data product
        self.b_span = data.MAGF_INST.data
        self.v_span = data.VEL_INST.data

        # Get the b-unit vector
        self.bvec = self.b_span/np.linalg.norm(self.b_span, axis=1)[:,NAX]

        # Get the angle between b and v.
        self.theta_bv = np.degrees(np.arccos(np.einsum('ij, ij->i', self.v_span, self.b_span)/(np.linalg.norm(self.v_span, axis=1) * np.linalg.norm(self.b_span, axis=1))))

        # storing the L2 and L3 times; we check later for each timestamp if they agree since
        # we want to use the correct magnetic field for its corresponding VDF.
        self.l3_time = data.Epoch.data
        self.l2_time = time

        # finding the N2D length scale for cartesian or hybrid
        self.kmax_arr = fn.find_kmax_arr(self, vdf_dict, self.Lmax)

        # the array of final TH
        self.TH_all = np.zeros(len(time))
        self.vshift_all = np.zeros(len(time))
        self.N2D_polcap_all = np.zeros(len(time))
        self.N2D_cart_all = np.zeros(len(time))

    def get_coors(self, u_bulk, tidx):
        r"""
        This function is used to setup the grids in gyrotropic coordinate depending on the 
        proposed gyrocenter and the induced shift along the :math:`v_{||}` direction
        (which depends on the polar cap extent). This is function is evaluated in every
        iteration of the MCMC code and whenever a new gyrocentroid is being proposed. The
        fitting (both in the polar cap and Cartesian frameworks) is performed in the 
        gyrotropic grid produced from this function.

        Parameters
        ----------
        u_bulk : array-like of floats
            The (3,) array containing the proposed bulk velocity in the instrument's (X,Y,Z) frame.

        tidx : int
            The time index being reconstructed.
        """
        self.vpara, self.vperp1, self.vperp2, self.vperp = None, None, None, None
        self.ubulk = u_bulk * 1.0
        
        # Shift into the plasma frame for the proposed u_bulk (the gyrocentroid)
        self.ux = self.vx[tidx] - u_bulk[0, NAX, NAX, NAX]
        self.uy = self.vy[tidx] - u_bulk[1, NAX, NAX, NAX]
        self.uz = self.vz[tidx] - u_bulk[2, NAX, NAX, NAX]

        # Rotate the plasma frame data into the magnetic field aligned frame.
        vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(self.ux, self.uy, self.uz,
                                                                        *fn.field_aligned_coordinates(self.b_span[tidx])))                                                            
        
        # converting to the field aligned coordinates here [BEFORE SHIFTING ALONG vpara]
        self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
        self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

        # Check angle between flow and magnetic field. 
        # NOTE: NEED TO CHECK THIS CALCULATION.
        if (self.theta_bv[tidx] < 90):
            self.vpara = -1.0 * self.vpara
            self.theta_sign = -1.0
        else: self.theta_sign = 1.0

        # Boosting along vpara [this step is crucial for the polar cap method, only]
        vpara1 = self.vpara - np.nanmax(self.vpara)
        max_r = np.nanmax(self.vperp[self.nanmask[tidx]]/np.tan(np.radians(self.TH)) + (vpara1[self.nanmask[tidx]]))
        self.vshift = max_r + np.nanmax(self.vpara)
        self.vpara -= self.vshift

        # converting the grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90

        # making the knots everytime the gyrotropic coordinates are changed
        self.make_knots(tidx)

    def get_coors_update_TH(self, u_bulk, tidx):
        r"""
        This function is used to setup the super-resolution grids in gyrotropic coordinate depending on the 
        proposed gyrocenter and the induced shift along the :math:`v_{||}` direction
        (which depends on the polar cap extent). The super-resolution
        (both in the polar cap and Cartesian frameworks) is performed in the 
        gyrotropic grid produced from this function.

        Parameters
        ----------
        u_bulk : array-like of floats
            The (3,) array containing the proposed bulk velocity in the instrument's (X,Y,Z) frame.

        tidx : int
            The time index being reconstructed.
        """
        self.vpara, self.vperp1, self.vperp2, self.vperp = None, None, None, None
        self.ubulk = u_bulk * 1.0
        
        # Shift into the plasma frame for the proposed u_bulk (the gyrocentroid)
        self.ux = self.vx[tidx] - u_bulk[0, NAX, NAX, NAX]
        self.uy = self.vy[tidx] - u_bulk[1, NAX, NAX, NAX]
        self.uz = self.vz[tidx] - u_bulk[2, NAX, NAX, NAX]

        # Rotate the plasma frame data into the magnetic field aligned frame.
        vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(self.ux, self.uy, self.uz,
                                                                        *fn.field_aligned_coordinates(self.b_span[tidx])))                                                            
        
        # converting to the field aligned coordinates here [BEFORE SHIFTING ALONG vpara]
        self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
        self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

        # Check angle between flow and magnetic field. 
        # NOTE: NEED TO CHECK THIS CALCULATION.
        if (self.theta_bv[tidx] < 90):
            self.vpara = -1.0 * self.vpara
            self.theta_sign = -1.0
        else: self.theta_sign = 1.0

        # Boosting along vpara [this step is crucial for the polar cap method, only]
        self.vshift = self.velocity[tidx, *self.max_indices[tidx]] + vpara[*self.max_indices[tidx]]
        self.vpara -= self.vshift
        self.vshift_all[tidx] = self.vshift * 1.0

        # converting the vpara shifted grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90

        # finding the maximum TH (comes from the maximum vperp) for polarcap Slepian generation
        self.TH = np.max(np.degrees(np.arctan2(self.vperp[self.nanmask[tidx]], -self.vpara[self.nanmask[tidx]])))
        self.TH_all[tidx] = self.TH * 1.0

        # generating the polcap Slepians for the new TH
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)

        # truncating beyond N2D; if None is passed it uses the default Shannon number definition
        self.N2D_polcap = int(np.sum(self.Slep.V))
        self.N2D_polcap_all[tidx] = int(np.sum(self.Slep.V))

        # generating the Slepian normalizations to be later used for Bspline regularization (in the D matrix)
        self.Slep.gen_Slep_norms()
        self.Slep.norm = self.Slep.norm[:self.N2D_polcap]

        # making the knots everytime the gyrotropic coordinates are changed
        self.make_knots(tidx)

    def make_knots(self, tidx):
        """
        Creates the knots in :math:`r = \sqrt{v_{||}^2 + v_{\perp}}` space. The default 
        spacing of the knots are assumed to be uniform in log-space. This is altered to
        accomodate a SPLINE_MINCOUNT number of grids for each knot (bspline). The final
        generated knots are in [km/s] space. This function is called everytime the gyro
        coordinates are changed due to a new proposed bulk velocity.

        Parameters
        ----------
        tidx : int
            The time index for which we generate the knots.
        """
        self.knots, self.vpara_nonan = None, None

        # the gyrotropic grids filtered by count threshold
        self.vpara_nonan = self.r_fa[self.nanmask[tidx]] * np.cos(np.radians(self.theta_fa[self.nanmask[tidx]]))
        self.vperp_nonan = self.r_fa[self.nanmask[tidx]] * np.sin(np.radians(self.theta_fa[self.nanmask[tidx]]))
        
        # this is used to create the knots for B-splines
        self.rfac_nonan = self.r_fa[self.nanmask[tidx]]

        # finding the minimum and maximum velocities with counts to find the knot locations
        vmin = np.min(self.velocity[tidx, self.nanmask[tidx]])
        vmax = np.max(self.velocity[tidx, self.nanmask[tidx]])
        
        # this is calculated from the mean(log10(v_{i+1}) - log10(v_i)) of SPAN-i grid
        self.dlnv = 2 * np.nanmean(np.diff(np.log10(self.velocity[tidx,:,0,0])))

        # making the initial estimate of counts per bin and bin edges in log space of knots 
        Nbins = int((np.log10(vmax) - np.log10(vmin)) / self.dlnv)
        counts, bin_edges = np.histogram(np.log10(self.rfac_nonan), bins=Nbins)
        
        # NOTE: ARE THESE TWO LINES NECESSARY? WHERE ARE THESE ATTRIBUTES BEING USED?
        gvdf_tstamp.hist_counts = counts
        gvdf_tstamp.bins = (bin_edges[0:-1] + bin_edges[1:])/2

        # merging bins to have spline_mincount number of counts per bin of knots
        new_edges, _ = fn.merge_bins(bin_edges, counts, self.spline_mincount)
        log_knots = np.sum(new_edges, axis=1)/2

        # adding the first and last points as knots
        log_knots = np.append(new_edges[0][0] - self.dlnv/2., log_knots)
        log_knots = np.append(log_knots, new_edges[-1][-1] + self.dlnv/2.)

        # converting to [km/s] units for the knot locations in velocity phase space
        self.knots = np.power(10, log_knots)

        # arranging the knots in an increasing order
        self.knots = np.sort(self.knots)

    def inversion(self, u_bulk, vdfdata, tidx):
        """
        Function that always performs the polcap inversion. Only used in the 
        iterative determination of the gyrocentroid. This is NOT used for final
        super-resolution. This is why the polap.inversion() is hard-coded here.

        Parameters
        ----------
        ubulk : array-like of float
            Array containing the proposed bulk velocity of shape (3,). 

        vdfdata : array-like of float
            The array for log10 of the SPAN-i VDF data scaled by the minimum value at that 
            time stamp (after filtering by count threshold).

        tidx : int
            The time index being reconstructed.

        Returns
        -------
        vdf_inv : array-like of floats
            The inferred VDF evaluated on the gyrotropic grids.
        
        zeromask : array-like of bools
            A mask which is True where the reconstructed VDF is zero.
            NOTE: I am not sure why we need this anymore.
        """ 
        # making the gyrotropic coordinates for the given u_bulk and magnetic field at that timestamp
        self.get_coors(u_bulk, tidx)

        # performing the inversion and obtaining the reconstructed VDF on the SPAN-i grids
        vdf_rec, zeromask = polcap.inversion(self, vdfdata, tidx)

        return vdf_rec, zeromask
    
    def super_resolution(self, u_bulk, tidx, NPTS):
        """
        Parameters
        ----------
        ubulk : array-like of float
            Array containing the most optimal choice of bulk velocity of shape (3,). 

        vdfdata : array-like of float
            The array for log10 of the SPAN-i VDF data scaled by the minimum value at that 
            time stamp (after filtering by count threshold).

        tidx : int
            The time index being super-resolved.

        NPTS : int
            The final super-resoluiton will be (NPTS x NPTS) in shape.

        Returns
        -------
        vdf_inv : array-like of floats
            The inferred VDF evaluated on the gyrotropic grids.
        
        zeromask : array-like of bools
            A mask which is True where the reconstructed VDF is zero.
            NOTE: I am not sure why we need this anymore.
        
        vdf_super : array-like of floats (only for super resolution)
            The super-resolved VDF.
        
        data_misfit : array-like of float (only for super resolution for polar cap)
            The data misfit array for different levels of regularization. This is only returned 
            passed during the super-resolution for the polar caps since the Cartesian fitting
            does not use a regularization. Also, this is only returned during super-resolution
            since we do not need a smooth estimate when determining the gyro-symmetry.
            NOTE: This is returned only to be able to make the L-curve plot for diagnostic purposes.
        
        model_misfit : array-like of float (only for super resolution for polar cap)
            The model misfit array for different levels of regularization. This is only returned 
            passed during the super-resolution for the polar caps since the Cartesian fitting
            does not use a regularization. Also, this is only returned during super-resolution
            since we do not need a smooth estimate when determining the gyro-symmetry.
            NOTE: This is returned only to be able to make the L-curve plot for diagnostic purposes.
        """
        # making the gyrotropic coordinates for the finalized u_bulk and magnetic field at that timestamp
        self.get_coors_supres(u_bulk, tidx)

        vdf_inv, vdf_super, zeromask, data_misfit, model_misfit = self.inversion_code.super_resolution(self, tidx, NPTS)

        return vdf_inv, vdf_super, zeromask, data_misfit, model_misfit

#---------------ALL FUNCTIONS IN THIS BLOCK ARE USED ONLY TO FIND THE GYROCENTROID--------------#
def log_prior_perpspace(model_params):
    Vperp1, Vperp2 = model_params
    if -100 < Vperp1 < 100 and -100 < Vperp2 < 100:
        return 0.0
    return -np.inf

def log_probability_perpspace(model_params, vdfdata, tidx, u_init_scipy, u, v):
    lp = log_prior_perpspace(model_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_perpspace(model_params, u_init_scipy, vdfdata, tidx, u, v)

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

#---------------------------------------------------------------------------------------------#
def main(START_INDEX = 0, NSTEPS = None, NPTS_SUPER=49,
         MCMC = False, MCMC_WALKERS=6, MCMC_STEPS=200,
         MIN_METHOD='L-BFGS-B', SAVE_FIGS=False, SAVE_PKL=True):
    """
    All the real computations happen in this function --- loading the data, finding the
    gyrocentroid and finding the final super-resolution. The input parameters are the same
    as those in the config file.
    """
    # the dictionary that is finally saved as a .pkl file
    vdf_rec_bundle = {}

    # computes till the last timestamp available if NSTEPS is not specified
    if(NSTEPS is None): NSTEPS = len(psp_vdf.time.data)

    # if a scalar is provided for NPTS_SUPER, convert it to a tuple
    if(isinstance(NPTS_SUPER, tuple)): pass
    else: NPTS_SUPER = (NPTS_SUPER, NPTS_SUPER)

    # storing the dimensions of super resolution
    gvdf_tstamp.nptsx, gvdf_tstamp.nptsy = NPTS_SUPER

    # the main time loop of GDF reconstruction
    for tidx in tqdm(range(START_INDEX, START_INDEX + NSTEPS)):
        # Check the l2 and l3 times match (to ensure selection of correct magnetic field)
        if gvdf_tstamp.l2_time[tidx] != gvdf_tstamp.l3_time[tidx]:
            print('Index mismatch. Skipping time.')
            continue

        # initializing the vdf data to optimize (this is the normalized and logarithmic value)
        vdfdata = np.log10(psp_vdf.vdf.data[tidx, gvdf_tstamp.nanmask[tidx]]/gvdf_tstamp.minval[tidx])
        gvdf_tstamp.vdfdata = vdfdata * 1.0

        # initializing the gyrocentroid to the reported SPAN velocity moment
        u_origin = gvdf_tstamp.v_span[tidx,:]

        # the 3D array of grid points in VX, VY, VZ of the SPAN grid
        threeD_points = np.vstack([gvdf_tstamp.vx[tidx][gvdf_tstamp.nanmask[tidx]],
                                   gvdf_tstamp.vy[tidx][gvdf_tstamp.nanmask[tidx]],
                                   gvdf_tstamp.vz[tidx][gvdf_tstamp.nanmask[tidx]]]).T

        # first estimate of correction to the SPAN-moment using scipy.optimize.minimize
        u_corr, __, u, v = find_symmetry_point(threeD_points, vdfdata, gvdf_tstamp.bvec[tidx],
                                               loss_fn_Slepians, tidx, origin=u_origin,
                                               MIN_METHOD=MIN_METHOD)
        # storing this to compare with the MCMC estimate, if needed, No other reason.
        u_corr_scipy = u_corr * 1.0
        
        # computing super-resolution and moments from the scipy corrected bulk velocity
        vdf_inv, vdf_super, __, data_misfit, model_misfit  =\
                                    gvdf_tstamp.super_resolution(u_corr, tidx, NPTS_SUPER)
        den, vel, Tcomps, Trace = fn.vdf_moments(gvdf_tstamp, vdf_super, tidx)

        # This tells us how far off our v_parallel is from the defined assumed v_parallel
        delta_v = vel - gvdf_tstamp.vshift

        # calculate a correction to the bulk velocity to account for the artificial shift (in polcap method)
        u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*u_corr, *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
        u_xnew, u_ynew, u_znew = fn.inverse_rotate_vector_field_aligned(*np.array([u_para - delta_v, u_perp1, u_perp2]), *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
        u_adj = np.array([u_xnew, u_ynew, u_znew])

        # if we want to further refine the estimate and obtain error bounds
        if(MCMC):
            # after the scipy correction, we start assuming (0,0) in the perpendicular phase space
            Vperp1, Vperp2 = 0.0, 0.0
            
            # setting up the MCMC walkers with small perturbations about the initialization point
            nwalkers = MCMC_WALKERS
            Vperp1_pos = np.random.rand(nwalkers) + Vperp1
            Vperp2_pos = np.random.rand(nwalkers) + Vperp2
            pos = np.array([Vperp1_pos, Vperp2_pos]).T

            # TODO: MAY CONVERT TO MULTIPROCESSING SETUP, IF NEEDED.
            sampler = emcee.EnsembleSampler(nwalkers, 2, log_probability_perpspace, args=(vdfdata, tidx, u_adj, u, v))
            sampler.run_mcmc(pos, MCMC_STEPS, progress=True)
            
            # extracting the MCMC chains
            flat_samples = sampler.get_chain(flat=True)
            
            # plotting the results of the emcee posterior distribution functions
            if SAVE_FIGS:
                labels = [r"$V_{\perp 1}$", r"$V_{\perp 2}$"]
                fig = corner.corner(flat_samples, labels=labels, show_titles=True)
                plt.tight_layout()
                plt.savefig(f'./Figures/mcmc_dists_polcap/emcee_ubulk_{tidx}.png')
                plt.close(fig)

            # computing the 50th quantile level vales in (vperp1, vperp2) space [along u, v vectors]
            vperp_12_corr = np.quantile(flat_samples,q=[0.5],axis=0).squeeze()

            # final MCMC corrected bulk velocity correction to the u_adj from minimize
            u_corr = u_adj + vperp_12_corr[0] * u + vperp_12_corr[1] * v

            # making the uncertainties from the covariance matrix
            vperp_12_covmat = np.cov(flat_samples.T)
            sigma_x = project_uncertainty(vperp_12_covmat, u, v, 'x')
            sigma_y = project_uncertainty(vperp_12_covmat, u, v, 'y')
            sigma_z = project_uncertainty(vperp_12_covmat, u, v, 'z')

            # computing super-resolution and moments from MCMC final correction
            vdf_inv, vdf_super, __, data_misfit, model_misfit =\
                                        gvdf_tstamp.super_resolution(u_corr, tidx, NPTS_SUPER)
            den, vel, Tcomps, Trace = fn.vdf_moments(gvdf_tstamp, vdf_super, tidx)

            # This tells us how far off our v_parallel is from the defined/assumed v_parallel
            delta_v = vel - gvdf_tstamp.vshift

            # calculate a correction to the bulk velocity to account for the artificial shift (in polcap method)
            u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*u_corr, *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
            u_xnew, u_ynew, u_znew = fn.inverse_rotate_vector_field_aligned(*np.array([u_para - delta_v, u_perp1, u_perp2]), *fn.field_aligned_coordinates(gvdf_tstamp.b_span[tidx]))
            u_adj = np.array([u_xnew, u_ynew, u_znew])

        # saving this for plotting of the polcap reconstruction
        gvdf_tstamp.vel = vel

        if(SAVE_FIGS): gvdf_tstamp.plotter_func(gvdf_tstamp, vdf_inv, vdf_super, tidx,
                                                model_misfit=model_misfit, data_misfit=data_misfit,
                                                GRID=True, SAVE_FIGS=SAVE_FIGS)

        # bundling the post-processed parameters of interest
        bundle = {}
        bundle['den'] = den
        print('Density:', den)
        bundle['time'] = gvdf_tstamp.l2_time[tidx]
        bundle['component_temp'] = Tcomps
        bundle['scalar_temp'] = Trace
        bundle['u_final'] = u_adj
        bundle['data_misfit'] = data_misfit
        bundle['model_misfit'] = model_misfit

        # saving additional parameters if MCMC is True
        if(MCMC):
            bundle['u_corr']  = u_corr
            bundle['u_corr_scipy'] = u_corr_scipy
            bundle['vperp_12_covmat'] = vperp_12_covmat
            bundle['sigma_x'] = sigma_x
            bundle['sigma_y'] = sigma_y
            bundle['sigma_z'] = sigma_z

        vdf_rec_bundle[tidx] = bundle

    if(SAVE_PKL):
        ts0 = datetime.strptime(str(gvdf_tstamp.l2_time[START_INDEX])[0:26], '%Y-%m-%dT%H:%M:%S.%f')
        ts1 = datetime.strptime(str(gvdf_tstamp.l2_time[START_INDEX + NSTEPS - 1])[0:26], '%Y-%m-%dT%H:%M:%S.%f')
        ymd = ts0.strftime('%Y%m%d')
        a_label = ts0.strftime('%H%M%S')
        b_label = ts1.strftime('%H%M%S')
        misc_fn.write_pickle(vdf_rec_bundle, f'./Outputs/scipy_vdf_rec_data_{MCMC_WALKERS}_{MCMC_STEPS}_{ymd}_{a_label}_{b_label}')

def run(config):
    """
    Primary function to drive the inversion code which simply takes in the config dictionary. This 
    function is meant to be used on the user end.

    Parameters
    ----------
    config : dictionary
        Dictionary containing the input parameters from the initialization file (e.g., init_gdf.py)

    Returns
    -------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for last timestamp being reconstructed. This is
        primarily for single-timestamp investigations of the class for debugging purposes.
    """
    global psp_vdf, gvdf_tstamp

    # loading the credentials from the file
    creds  = misc_fn.credential_reader(config['global']['CREDS_PATH'])

    # loading the PSP data for the given TRANGE with optional clipping
    psp_vdf = fn.init_psp_vdf(config['global']['TRANGE'], CREDENTIALS=creds, CLIP=config['global']['CLIP'])

    # initializing the gvdf_tstamp class
    gvdf_tstamp = gyrovdf(psp_vdf, config, CREDENTIALS=creds)

    main(START_INDEX=config['global']['START_INDEX'],
         NSTEPS=config['global']['NSTEPS'],
         NPTS_SUPER=config['global']['NPTS_SUPER'],
         MCMC=config['global']['MCMC'],
         MCMC_WALKERS=config['global']['MCMC_WALKERS'],
         MCMC_STEPS=config['global']['MCMC_STEPS'],
         MIN_METHOD=config['global']['MIN_METHOD'],
         SAVE_FIGS=config['global']['SAVE_FIGS'],
         SAVE_PKL =config['global']['SAVE_PKL'])

    return gvdf_tstamp

if __name__=='__main__':
    config_file = sys.argv[1]
    config = importlib.import_module(config_file)
    gvdf_tstamp = run(config)