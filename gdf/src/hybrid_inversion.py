import numpy as np
from scipy.linalg import solve

from gdf.src import polar_cap_inversion as polcap_inversion
from gdf.src import functions as fn
from gdf.src import basis_funcs as basis_fn
from gdf.src import misc_funcs as misc_fn

def find_polcap_knee_idx(gvdf_tstamp):
    #--------- starting the inversion for obtaining the coefficients using B-spline regularization-------#
    G_g = gvdf_tstamp.G_k_n @ gvdf_tstamp.G_k_n.T
    
    # Setup the BSpline regularization
    D_i_i = basis_fn.get_Bspline_second_derivative(gvdf_tstamp.knots, gvdf_tstamp.p, gvdf_tstamp.super_vpara)

    # Augment the D-matrix
    gvdf_tstamp.D = np.kron(D_i_i, np.diag(gvdf_tstamp.Slep.norm))
    I = np.identity(len(G_g))

    # making the data and model misfit arrays
    data_misfit, model_misfit = [], []

    for mu in gvdf_tstamp.mu_arr:
        # coeffs = np.linalg.inv(G_g + mu * gvdf_tstamp.D) @ gvdf_tstamp.G_k_n @ vdfdata
        coeffs = solve(G_g + mu * gvdf_tstamp.D, gvdf_tstamp.G_k_n @ gvdf_tstamp.vdfdata, assume_a='sym')

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

def find_hybrid_knee_idx(gvdf_tstamp, G, d, hybrid_dict):
    #--------- starting the inversion for obtaining the coefficients using similarity regularization-------#
    # making the data and model misfit arrays
    data_misfit, model_misfit = [], []

    for lam in gvdf_tstamp.lambda_arr:
        # adding the instantaneous lambda to perform the inversion
        G_lam = G * 1.0
        G_lam[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf'],
              :hybrid_dict['nparams_A']] *= np.sqrt(lam)
        G_lam[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf'],
          hybrid_dict['nparams_A']:hybrid_dict['nparams_A']+hybrid_dict['nparams_B']] *= np.sqrt(lam)

        # coeffs = np.linalg.inv(G_g + mu * gvdf_tstamp.D) @ gvdf_tstamp.G_k_n @ vdfdata
        coeffs = solve(G_lam.T @ G_lam, G_lam.T @ d, assume_a='sym')

        # slicing out the respective coefficients 
        m_polcap, m_cartesian = coeffs[:hybrid_dict['nparams_A']], coeffs[hybrid_dict['nparams_A']:]

        # reconstructing the two models on the data grids
        vdf_inv_polcap = hybrid_dict['A'] @ m_polcap
        vdf_inv_cartesian = hybrid_dict['B'] @ m_cartesian

        # super-resolving the two models
        vdf_super_polcap = hybrid_dict['Af'] @ m_polcap
        vdf_super_cartesian = hybrid_dict['Bf'] @ m_cartesian

        # computing and appending the data misfit
        data_misfit.append(np.linalg.norm(vdf_inv_polcap - gvdf_tstamp.vdfdata)**2 + np.linalg.norm(vdf_inv_cartesian[hybrid_dict['ndata_A']:] - gvdf_tstamp.vdfdata)**2)

        # computing and appending the model misfit
        model_misfit.append(np.linalg.norm(vdf_super_polcap - vdf_super_cartesian)**2)

    # normalizing the misfit arrays for better knee finding
    model_misfit = misc_fn.norm_array(model_misfit)
    data_misfit = misc_fn.norm_array(data_misfit)

    # finding the knee of the L-curve and plotting, if necessary
    gvdf_tstamp.lambda_knee_idx = fn.geometric_knee(model_misfit, data_misfit)

    return data_misfit, model_misfit

def create_hybrid_Gmatrix(gvdf_tstamp, hybrid_dict):
    """
    Calculates the G matrix for performing the hybrid inversion where we impose the similarity index between the polcap and cartesian methods.

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.
    
    hybrid_dict : dictionary
        Contains all the components for building the augmented G matrix. This involves the data resolution
        matrices for the polcap and Cartesian methods as well as the same matrices on the super-resolution grids.

    Returns
    -------
    G : array_like of floats
        The G matrix for performing hybrid inversion.
    """
    # creating the G matrix
    G = np.zeros((hybrid_dict['ndata_A'] + hybrid_dict['ndata_B'] + hybrid_dict['nf'] + hybrid_dict['nparams_A'],
                  hybrid_dict['nparams_A'] + hybrid_dict['nparams_B']))

    # filling in A responsible for polar cap reconstruction
    G[:hybrid_dict['ndata_A'], :hybrid_dict['nparams_A']] += hybrid_dict['A']
    # filling in B responsible for Cartesian reconstruction
    G[hybrid_dict['ndata_A']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B'],
      hybrid_dict['nparams_A']:hybrid_dict['nparams_A']+hybrid_dict['nparams_B']] = hybrid_dict['B']

    # filling in the (A(m1) - B(m2)) term ---> to ensure similarity between the two reconstructions
    G[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf'],
      :hybrid_dict['nparams_A']] = hybrid_dict['Af']   # the \sqrt(lambda) will be multiplied later
    G[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf'],
      hybrid_dict['nparams_A']:hybrid_dict['nparams_A']+hybrid_dict['nparams_B']] = -hybrid_dict['Bf']   # the \sqrt(lambda) will be multiplied later
    
    # finding the un-contracted regularization matrix P such that D = P.T @ P
    eigvals, eigvecs = np.linalg.eigh(gvdf_tstamp.D)
    sqrt_eigvals = np.sqrt(np.clip(eigvals, 0, None))  # Ensure non-negative
    P_reg = np.diag(sqrt_eigvals) @ eigvecs.T

    # including the B-spline regularization for polar-cap model
    G[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf']+hybrid_dict['nparams_A'],
      :hybrid_dict['nparams_A']] = -np.sqrt(gvdf_tstamp.mu_arr[gvdf_tstamp.knee_idx]) * P_reg

    # creating the augmented data matrix
    d = np.zeros((hybrid_dict['ndata_A'] + hybrid_dict['ndata_B'] + hybrid_dict['nf'] + hybrid_dict['nparams_A']))
    d[:hybrid_dict['ndata_A']] = gvdf_tstamp.vdfdata
    # the SPAN-i data to be used
    vdf_data = np.append(gvdf_tstamp.vdfdata, gvdf_tstamp.vdfdata)
    d[hybrid_dict['ndata_A']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']] = vdf_data

    return G, d

def inversion_hybrid(gvdf_tstamp, hybrid_dict):
    """
    Calculates the coefficients for both the polcap and cartesian reconstructions by imposing 
    a degree of similarity between them using the LAMBDA parameter specified in the initialization file.

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.
    
    hybrid_dict : dictionary
        Contains all the components for building the augmented G matrix. This involves the data resolution
        matrices for the polcap and Cartesian methods as well as the same matrices on the super-resolution grids.

    Returns
    -------
    m_polar : array-like of floats
        1D array of coefficients for the polar cap inversion (similarity induced).

    m_cartesian : array-like of floats
        1D array of coefficients for the cartesian inversion (similarity induced).
    """
    # generating the G and d matrices for hybrid inversion
    G, d = create_hybrid_Gmatrix(gvdf_tstamp, hybrid_dict)

    # finding the knee index for implementing B-spline regularization
    if(gvdf_tstamp.lam is None): 
        # finding the knee of the trade-off curve
        data_misfit, model_misfit = find_hybrid_knee_idx(gvdf_tstamp, G, d, hybrid_dict)
        gvdf_tstamp.lambda_knee = gvdf_tstamp.lambda_arr[gvdf_tstamp.lambda_knee_idx]
    else:
        data_misfit, model_misfit = None, None
        gvdf_tstamp.lambda_knee = gvdf_tstamp.lam * 1.0

    # now adding the lambda corresponding to the knee of the trade-off curve
    G[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf'],
     :hybrid_dict['nparams_A']] *= np.sqrt(gvdf_tstamp.lambda_knee)
    G[hybrid_dict['ndata_A']+hybrid_dict['ndata_B']:hybrid_dict['ndata_A']+hybrid_dict['ndata_B']+hybrid_dict['nf'],
      hybrid_dict['nparams_A']:hybrid_dict['nparams_A']+hybrid_dict['nparams_B']] *= np.sqrt(gvdf_tstamp.lambda_knee)

    # calculating the coefficients (m_polcap + m_cartesian)
    GTG = G.T @ G
    GTd = G.T @ d 
    # the 'sym' option is used since we know GTG is a symmetric matrix
    m = solve(GTG, GTd, assume_a='sym')

    # slicing out the respective coefficients 
    m_polcap, m_cartesian = m[:hybrid_dict['nparams_A']], m[hybrid_dict['nparams_A']:]

    return m_polcap, m_cartesian, data_misfit, model_misfit

def super_resolution(gvdf_tstamp, tidx, NPTS):    
    """
    Uses polar cap and Cartesian Slepians generated inside the convex hull to super-resolve the GDF
    for a given timestamp. Imposes a similarity between the reconstruction based on the LAMBDA 
    parameter specified in the initialization file.

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.
        Should already contain the CartSlep class instance generated as an attribute.
    
    tidx : int
        The time index being super-resolved.

    Returns
    -------
    This returns a total of r parameters but only the first two are valid. This is done to keep with 
    the same convention as the polcar_cap_inverion.super_resolution().

    vdf_inv : array-like of floats
        The inferred VDF evaluated on the SPAN data grids in the form of (vdf_inv_polcap, vdf_inv_cartesian).
    
    vdf_super : array-like of floats (only for super resolution)
        The super-resolved  GDF of shape (gvdf_stamp.nptsx, gvdf_tstamp.nptsy). Domain extent is 
        automatically determined from the convex hull boundary. This is also returned in the form
        (vdf_super_polcap, vdf_super_cartesian).
    """
    # this is populated at each timestamp inversion
    hybrid_dict = {}

    # setting up grids, boundaries and hull for hybrid super-resolution
    polcap_inversion.define_supres_polgrids(gvdf_tstamp, NPTS)

    # calculating the N2D for hybrid if it is None (not user-specified)
    if(gvdf_tstamp.N2D_cart_default is None): 
        gvdf_tstamp.N2D_cart = np.min([fn.find_N2D_cart(gvdf_tstamp, tidx), gvdf_tstamp.N2D_cart_max])
    else:
        gvdf_tstamp.N2D_cart = gvdf_tstamp.N2D_cart_default * 1.0

    gvdf_tstamp.N2D_cart_all[tidx] = gvdf_tstamp.N2D_cart * 1.0

    #---------------------------------POLCAP SETUP 1----------------------------------#
    # creating the B-splines, Slepian functions (at new theta grids) and G matrix about the finalized ubulk
    polcap_inversion.get_Bsplines(gvdf_tstamp)
    polcap_inversion.get_Slepians(gvdf_tstamp, tidx)
    polcap_inversion.get_G_matrix(gvdf_tstamp)

    find_polcap_knee_idx(gvdf_tstamp)

    #-------------------------------CARTESIAN SETUP 1----------------------------------#
    # getting the Slepians on the measurement points
    gvdf_tstamp.CartSlep.gen_Slep_basis(gvdf_tstamp.boundary_points, np.double(gvdf_tstamp.N2D_cart),
                                        np.array([gvdf_tstamp.v_perp_all, gvdf_tstamp.v_para_all]).T)

    # storing the data resolution matrices and parameters [will be used in creating the augmented matrix]
    hybrid_dict['A'] = gvdf_tstamp.G_k_n.T
    hybrid_dict['B'] = gvdf_tstamp.CartSlep.G
    hybrid_dict['ndata_A'], hybrid_dict['nparams_A'] = hybrid_dict['A'].shape
    hybrid_dict['ndata_B'], hybrid_dict['nparams_B'] = hybrid_dict['B'].shape

    #----------------------------------POLCAP SETUP 2----------------------------------#
    # creating the new B-splines, Slepian functions (at new theta grids) and G matrix [for SUPERRESOLUTION]
    polcap_inversion.super_Bsplines(gvdf_tstamp)
    polcap_inversion.super_Slepians(gvdf_tstamp)
    polcap_inversion.super_G_matrix(gvdf_tstamp)

    #-------------------------------CARTESIAN SETUP 2----------------------------------#
    # getting the Slepians on the super-resolution points
    gvdf_tstamp.CartSlep.gen_Slep_basis(gvdf_tstamp.boundary_points, np.double(gvdf_tstamp.N2D_cart),
                                        gvdf_tstamp.grid_points)

    # storing the super-resolution matrices [will be used in creating the augmented matrix]
    hybrid_dict['Af'] = gvdf_tstamp.super_G_k_n.T
    hybrid_dict['Bf'] = gvdf_tstamp.CartSlep.G
    hybrid_dict['nf'] = hybrid_dict['Af'].shape[0]

    # transposing Bf for grid compatibility
    AfT = np.reshape(hybrid_dict['Af'], (int(np.sqrt(hybrid_dict['nf'])),
                                         int(np.sqrt(hybrid_dict['nf'])), 
                                         hybrid_dict['nparams_A']))
    AfT = np.transpose(AfT, [1,0,2])
    hybrid_dict['Af'] = np.reshape(AfT, (-1, hybrid_dict['nparams_A']))

    # transposing Bf for grid compatibility
    BfT = np.reshape(hybrid_dict['Bf'], (int(np.sqrt(hybrid_dict['nf'])),
                                         int(np.sqrt(hybrid_dict['nf'])), 
                                         hybrid_dict['nparams_B']))
    BfT = np.transpose(BfT, [1,0,2])
    hybrid_dict['Bf'] = np.reshape(BfT, (-1, hybrid_dict['nparams_B']))

    # calculating the coefficients
    m_polcap, m_cartesian, data_misfit, model_misfit = inversion_hybrid(gvdf_tstamp, hybrid_dict)

    # reconstructing the two models on the data grids
    vdf_inv_polcap = hybrid_dict['A'] @ m_polcap
    vdf_inv_cartesian = hybrid_dict['B'] @ m_cartesian
    vdf_inv = (vdf_inv_polcap, vdf_inv_cartesian)

    # super-resolving the two models
    vdf_super_polcap = hybrid_dict['Af'] @ m_polcap
    vdf_super_cartesian = hybrid_dict['Bf'] @ m_cartesian
    vdf_super = (vdf_super_polcap, vdf_super_cartesian)

    return vdf_inv, vdf_super, None, data_misfit, model_misfit