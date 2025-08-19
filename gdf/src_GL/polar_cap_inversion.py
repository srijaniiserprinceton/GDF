import numpy as np
from scipy.linalg import solve
NAX = np.newaxis

from gdf.src_GL import functions as fn
from gdf.src_GL import basis_funcs as basis_fn
from gdf.src_GL import misc_funcs as misc_fn
from gdf.src_GL import quadrature

def get_Bsplines_inst(gvdf_tstamp):
    gvdf_tstamp.B_i_n = basis_fn.get_Bsplines_scipy(gvdf_tstamp.knots_inst, gvdf_tstamp.p, gvdf_tstamp.rfac_nonan_inst)

def get_Bsplines_GL(gvdf_tstamp):
    gvdf_tstamp.B_i_n = basis_fn.get_Bsplines_scipy(gvdf_tstamp.knots_inst, gvdf_tstamp.p, gvdf_tstamp.rfac_nonan_GL)
   
def get_Slepians_inst(gvdf_tstamp, tidx):
    gvdf_tstamp.S_alpha_n = basis_fn.get_Slepians_scipy(gvdf_tstamp.Slep.C, gvdf_tstamp.theta_nonan_inst, 
                                                        gvdf_tstamp.Lmax, gvdf_tstamp.N2D_polcap)   

def get_Slepians_GL(gvdf_tstamp, tidx):
    gvdf_tstamp.S_alpha_n = basis_fn.get_Slepians_scipy(gvdf_tstamp.Slep.C, gvdf_tstamp.theta_nonan_GL, 
                                                        gvdf_tstamp.Lmax, gvdf_tstamp.N2D_polcap)
    
def get_G_matrix_inst(gvdf_tstamp):
    gvdf_tstamp.G_k_n = None
    gvdf_tstamp.G_i_alpha_n = None

    # taking the product to make the shape (i x alpha x n)
    gvdf_tstamp.G_i_alpha_n = gvdf_tstamp.B_i_n[:,NAX,:] * gvdf_tstamp.S_alpha_n[NAX,:,:]

    # flattening the k=(i, alpha) dimension to make the shape (k x n)
    npoints = len(gvdf_tstamp.vpara_nonan_inst)
    gvdf_tstamp.G_k_n = np.reshape(gvdf_tstamp.G_i_alpha_n, (-1, npoints))
    
    gvdf_tstamp.G_k_n_notavg = gvdf_tstamp.G_k_n * 1.0

def get_G_matrix_GL(gvdf_tstamp, tidx):
    gvdf_tstamp.G_k_n = None
    gvdf_tstamp.G_i_alpha_n = None

    # taking the product to make the shape (i x alpha x n)
    G_i_alpha_nnq_GL = gvdf_tstamp.B_i_n[:,NAX,:] * gvdf_tstamp.S_alpha_n[NAX,:,:]
    Nnq = G_i_alpha_nnq_GL.shape[-1]

    # flattening the k=(i, alpha) dimension to make the shape (k x n)
    G_k_nnq_GL = np.reshape(G_i_alpha_nnq_GL, (-1, Nnq))

    # unflattening the Npoints_inst and Nquadrature dimensions
    npoints_inst = len(gvdf_tstamp.vpara_nonan_inst)
    G_k_n_nq_GL = np.reshape(G_k_nnq_GL, (-1, npoints_inst, gvdf_tstamp.NQ_V * gvdf_tstamp.NQ_T * gvdf_tstamp.NQ_P), 'C')

    # this now has the same shape as the inst cell-centered grid measurements
    gvdf_tstamp.G_k_n = quadrature.GL_vol_avg_polcap(G_k_n_nq_GL, gvdf_tstamp, tidx)

def inversion_inst(gvdf_tstamp, vdfdata, tidx):
    """
    This function peforms the *on-the-SPAN-grid* inversion, with the primary objective
    of being used to find the gyro-centroid. The actual inversion with appropriate regularization
    is done in super_resolution().

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.

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
    # creating the new B-splines, Slepian functions (at new theta grids) and G matrix
    get_Bsplines_inst(gvdf_tstamp)
    get_Slepians_inst(gvdf_tstamp, tidx)
    get_G_matrix_inst(gvdf_tstamp)

    # obtaining the coefficients
    G_g = gvdf_tstamp.G_k_n @ gvdf_tstamp.G_k_n.T
    I = np.identity(len(G_g))
    G_d = gvdf_tstamp.G_k_n @ vdfdata
    # coeffs = np.linalg.inv(G_g + 1e-3 * I) @ G_d
    coeffs = solve(G_g + 1e-3 * I, G_d, assume_a='sym')

    # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
    vdf_rec = coeffs @ gvdf_tstamp.G_k_n

    # finding the zeros which need to be masked to avoid bad cost functions
    zeromask = vdf_rec == 0

    return vdf_rec, zeromask

def inversion_GL(gvdf_tstamp, vdfdata, tidx):
    """
    This function peforms the *on-the-SPAN-grid* inversion, with the primary objective
    of being used to find the gyro-centroid. The actual inversion with appropriate regularization
    is done in super_resolution().

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.

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
    # creating the new B-splines, Slepian functions (at new theta grids) and G matrix
    get_Bsplines_inst(gvdf_tstamp)
    get_Slepians_GL(gvdf_tstamp, tidx)
    get_G_matrix_GL(gvdf_tstamp, tidx)

    # obtaining the coefficients
    G_g = gvdf_tstamp.G_k_n @ gvdf_tstamp.G_k_n.T
    I = np.identity(len(G_g))
    G_d = gvdf_tstamp.G_k_n @ vdfdata
    # coeffs = np.linalg.inv(G_g + 1e-3 * I) @ G_d
    coeffs = solve(G_g + 1e-3 * I, G_d, assume_a='sym')

    # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
    vdf_rec = coeffs @ gvdf_tstamp.G_k_n

    # finding the zeros which need to be masked to avoid bad cost functions
    zeromask = vdf_rec == 0

    return vdf_rec, zeromask

def define_supres_polgrids(gvdf_tstamp, NPTS, plothull=False):
    # this is super-resolution grid in x
    gvdf_tstamp.v_perp_all = np.concatenate([-gvdf_tstamp.vperp_nonan_inst, gvdf_tstamp.vperp_nonan_inst])
    # this is super-resolution grid in y
    gvdf_tstamp.v_para_all = np.concatenate([gvdf_tstamp.vpara_nonan_inst, gvdf_tstamp.vpara_nonan_inst])

    # extracting the convex hull boundary
    supres_gridy_1D, supres_grids, boundary_points, hull_mask, area =\
                    fn.find_supres_grid_and_boundary(gvdf_tstamp.v_perp_all, gvdf_tstamp.v_para_all,
                                                     NPTS, plothull=plothull)
    
    # storing these as attributes of the class
    # to be used in masked moment calculations and generating Cartesian Slepians
    gvdf_tstamp.super_vpara = supres_gridy_1D * 1.0
    gvdf_tstamp.grid_points = supres_grids * 1.0
    gvdf_tstamp.boundary_points = boundary_points
    gvdf_tstamp.hull_mask = hull_mask
    gvdf_tstamp.hull_area = area

    gvdf_tstamp.super_rfac  = np.sqrt(gvdf_tstamp.grid_points[:,0]**2 + gvdf_tstamp.grid_points[:,1]**2) 
    gvdf_tstamp.super_theta = np.degrees(np.arctan2(gvdf_tstamp.grid_points[:,0], gvdf_tstamp.grid_points[:,1])) # stick to convention

def super_Bsplines(gvdf_tstamp):
    gvdf_tstamp.super_B_i_n = basis_fn.get_Bsplines_scipy(gvdf_tstamp.knots_inst, gvdf_tstamp.p, gvdf_tstamp.super_rfac)

def super_Slepians(gvdf_tstamp):
    gvdf_tstamp.super_S_alpha_n = basis_fn.get_Slepians_scipy(gvdf_tstamp.Slep.C, gvdf_tstamp.super_theta, 
                                                              gvdf_tstamp.Lmax, gvdf_tstamp.N2D_polcap)

def super_G_matrix(gvdf_tstamp):
    gvdf_tstamp.super_G_k_n = None
    gvdf_tstamp.super_G_i_alpha_n = None
    # taking the product to make the shape (i x alpha x n)
    gvdf_tstamp.super_G_i_alpha_n = gvdf_tstamp.super_B_i_n[:,NAX,:] * gvdf_tstamp.super_S_alpha_n[NAX,:,:]

    # flattening the k=(i, alpha) dimension to make the shape (k x n)
    npoints = len(gvdf_tstamp.super_rfac)
    gvdf_tstamp.super_G_k_n = np.reshape(gvdf_tstamp.super_G_i_alpha_n, (-1, npoints))

def super_resolution(gvdf_tstamp, tidx, NPTS):
    """
    Performs the final super-resolution including the regularization in B-splines. The 
    suepr-resolution values are calculated on the provided (grid_x, grid_y).

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.

    vdfdata : array-like of float
        The array for log10 of the SPAN-i VDF data scaled by the minimum value at that 
        time stamp (after filtering by count threshold).
    
    tidx : int
        The time index being reconstructed.

    NPTS : tuple of ints
        The final super-resoluiton will be (NPTS_x, NPTS_y) in shape.

    Returns
    -------
    vdf_inv : array-like of floats
        The inferred VDF evaluated on the gyrotropic grids.
    
    vdf_super : array-like of floats (only for super resolution)
        The super-resolved VDF.

    zeromask : array-like of bools
        A mask which is True where the reconstructed VDF is zero.
        NOTE: I am not sure why we need this anymore.
    
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
    
    knee_idx : float (only for super resolution for polar cap)
        The knee index of the regularization constant which is used for the polar cap
        reconstruction. This is determined from the knee detection algorithm performed in 
        geometric_knee() in functions.py.
    """
    # creating the B-splines, Slepian functions (at new theta grids) and G matrix about the finalized ubulk
    get_Bsplines_GL(gvdf_tstamp)
    get_Slepians_GL(gvdf_tstamp, tidx)
    get_G_matrix_GL(gvdf_tstamp, tidx)

    # converting the Cartesian regular grid to the polar coordinates for the polcap super-resolution
    define_supres_polgrids(gvdf_tstamp, NPTS)

    # creating the new B-splines, Slepian functions (at new theta grids) and G matrix [for SUPERRESOLUTION]
    super_Bsplines(gvdf_tstamp)
    super_Slepians(gvdf_tstamp)
    super_G_matrix(gvdf_tstamp)

    #--------- starting the inversion for obtaining the coefficients using B-spline regularization-------#
    G_g = gvdf_tstamp.G_k_n @ gvdf_tstamp.G_k_n.T
    
    # Setup the BSpline regularization
    D_i_i = basis_fn.get_Bspline_second_derivative(gvdf_tstamp.knots_inst, gvdf_tstamp.p, gvdf_tstamp.super_vpara)

    # Augment the D-matrix
    gvdf_tstamp.D = np.kron(D_i_i, np.diag(gvdf_tstamp.Slep.norm))
    I = np.identity(len(G_g))

    # making the data and model misfit arrays
    data_misfit, model_misfit = [], []

    for mu in gvdf_tstamp.mu_arr:
        # coeffs = np.linalg.inv(G_g + mu * gvdf_tstamp.D) @ gvdf_tstamp.G_k_n @ vdfdata
        coeffs = solve(G_g + mu * gvdf_tstamp.D, gvdf_tstamp.G_k_n  @ gvdf_tstamp.vdfdata, assume_a='sym')

        # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
        vdf_rec = coeffs @ gvdf_tstamp.G_k_n

        # computing and appending the data misfit
        data_misfit.append(np.linalg.norm(vdf_rec - gvdf_tstamp.vdfdata)**2)

        # computing and appending the model misfit
        model_misfit.append(np.linalg.norm(coeffs @ gvdf_tstamp.D @ coeffs)**2)

    # normalizing the misfit arrays for better knee finding
    model_misfit = misc_fn.norm_array(model_misfit)
    data_misfit = misc_fn.norm_array(data_misfit)

    # finding the knee of the L-curve and plotting, if necessary
    gvdf_tstamp.knee_idx = fn.geometric_knee(model_misfit, data_misfit)

    #------these are the final coefficients with the optimal (knee) choice for the regularization----------#
    # # coeffs = np.linalg.inv(G_g + gvdf_tstamp.mu_arr[knee_idx] * gvdf_tstamp.D) @ gvdf_tstamp.G_k_n @ gvdf_tstamp.vdfdata
    # coeffs = solve(G_g + gvdf_tstamp.mu_arr[gvdf_tstamp.knee_idx] * gvdf_tstamp.D,
    #                gvdf_tstamp.G_k_n @ gvdf_tstamp.vdfdata, assume_a='sym')
    coeffs = solve(G_g + gvdf_tstamp.mu_arr[gvdf_tstamp.knee_idx] * gvdf_tstamp.D,
                   gvdf_tstamp.G_k_n @ gvdf_tstamp.vdfdata, assume_a='sym')

    # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
    vdf_rec = coeffs @ gvdf_tstamp.G_k_n

    M = np.sum(np.power(10, gvdf_tstamp.vdfdata))
    I0 = np.sum(np.power(10, vdf_rec))
    print((M - I0) / M)

    # the superresolved VDF using the coefficient inferred from the sparse measurements
    vdf_super = coeffs.flatten() @ gvdf_tstamp.super_G_k_n

    # finding the zeros which need to be masked to avoid bad cost functions
    zeromask = vdf_rec == 0

    return vdf_rec, vdf_super, zeromask, data_misfit, model_misfit