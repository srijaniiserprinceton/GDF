import numpy as np
from scipy.linalg import solve
from line_profiler import profile

from gdf.src import eval_Slepians
from gdf.src import plotter
from gdf.src import functions as fn

def define_supres_cartgrids(gvdf_tstamp, NPTS, plothull=False):
    # this is super-resolution grid in x
    gvdf_tstamp.v_perp_all = np.concatenate([-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vperp_nonan])
    # this is super-resolution grid in y
    gvdf_tstamp.v_para_all = np.concatenate([gvdf_tstamp.vpara_nonan, gvdf_tstamp.vpara_nonan])

    # extracting the convex hull boundary
    supres_gridy_1D, supres_grids, boundary_points, hull_mask =\
                    fn.find_supres_grid_and_boundary(gvdf_tstamp.v_perp_all, gvdf_tstamp.v_para_all,
                                                     NPTS, plothull=plothull)
    
    # storing these as attributes of the class
    # to be used in masked moment calculations and generating Cartesian Slepians
    gvdf_tstamp.super_vpara = supres_gridy_1D * 1.0
    gvdf_tstamp.grid_points = supres_grids * 1.0
    gvdf_tstamp.boundary_points = boundary_points
    gvdf_tstamp.hull_mask = hull_mask

def inversion_CartSlep(gvdf_tstamp):
    """
    Computes the coefficients of the Cartesian Slepians from the SPAN-i data grid. Evaluates
    the low resolution Cartesian Slepians only.

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.
        Should already contain the CartSlep class instance generated as an attribute.

    Returns
    -------
    coeffs : array-like of floats
        The coefficients of the Cartesian Slepians evaluated from the linear inverse problem
        using the SPAN-i data grid values. These are used for super-resolution.
    """
    # getting the Slepians on the measurement points
    gvdf_tstamp.CartSlep.gen_Slep_basis(gvdf_tstamp.boundary_points, np.double(gvdf_tstamp.N2D_cart),
                                        np.array([gvdf_tstamp.v_perp_all, gvdf_tstamp.v_para_all]).T)

    # clipping off at the Shannon number
    gvdf_tstamp.CartSlep.G = gvdf_tstamp.CartSlep.G[:,:None]
    gvdf_tstamp.CartSlep.H = gvdf_tstamp.CartSlep.H[:,:None]

    # the data we intend to fit to
    vdf_data = np.append(gvdf_tstamp.vdfdata, gvdf_tstamp.vdfdata)

    # performing the inversion to get the Coefficients
    GTG = gvdf_tstamp.CartSlep.G.T @ gvdf_tstamp.CartSlep.G
    GTd = gvdf_tstamp.CartSlep.G.T @ vdf_data
    # the 'sym' option is used since we know GTG is a symmetric matrix
    coeffs = solve(GTG, GTd, assume_a='sym')

    return coeffs

def super_resolution(gvdf_tstamp, tidx, NPTS, plotSlep=False):
    """
    Uses Cartesian Slepians generated inside the convex hull to super-resolve the GDF
    for a given timestamp. Optionally plots the Cartesian Slepian basis functions for debugging.

    Parameters
    ----------
    gvdf_tstamp : gyrovdf class instance
        This is the class instance which is setup for the specific timestamp being reconstructed.
        Should already contain the CartSlep class instance generated as an attribute.
    
    tidx : int
        The time index being super-resolved.

    plotSlep : bool (optional)
        Flag to optionally plot the Cartsian Slepian basis functions, for diagnostic purposes only.

    Returns
    -------
    vdf_super : array-like of float
        Super-resolved GDF of shape (gvdf_stamp.nptsx, gvdf_tstamp.nptsy). Domain extent is 
        automatically determined from the convex hull boundary. 
    """
    # setting up grids, boundaries and hull for Cartesian super-resolution
    define_supres_cartgrids(gvdf_tstamp, NPTS)

    # inferring the coefficients from the data
    coeffs = inversion_CartSlep(gvdf_tstamp)
    # reconstruction on the SPAN-i data grid for comparison with vdfdata
    vdf_rec = coeffs @ gvdf_tstamp.CartSlep.G[:,:None].T

    # getting the Slepians on the super-resolution points
    gvdf_tstamp.CartSlep.gen_Slep_basis(gvdf_tstamp.boundary_points, np.double(gvdf_tstamp.N2D_cart),
                                        gvdf_tstamp.grid_points)

    # constructing the super-resolved grid from the coefficients of the data inversion
    vdf_super = coeffs @ gvdf_tstamp.CartSlep.G[:,:None].T

    # plotting the basis functions for diagnostic purposes
    if(plotSlep):
        xx = np.reshape(gvdf_tstamp.grid_points[:,0], (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy), 'F')
        yy = np.reshape(gvdf_tstamp.grid_points[:,1], (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy), 'F')
        plotter.plot_CartSlep(xx, yy, gvdf_tstamp.CartSlep, tidx)

    return vdf_rec, None, vdf_super, None, None