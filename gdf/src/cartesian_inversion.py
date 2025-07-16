import numpy as np
from line_profiler import profile

from gdf.src import eval_Slepians
from gdf.src import functions as fn
from gdf.src import plotter

def inversion_CartSlep(gvdf_tstamp, N):
    # setting up the inversion process
    eval_gridx = np.append(-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vperp_nonan)
    eval_gridy = np.append(gvdf_tstamp.vpara_nonan, gvdf_tstamp.vpara_nonan)

    # extracting the convex hull boundary
    boundary_points = fn.find_convexhull_boundary(gvdf_tstamp.v_para_all,
                                                  gvdf_tstamp.v_perp_all,
                                                  plothull=False)

    # getting the Slepians on the measurement points
    CartSlep = eval_Slepians.Slep_2D_Cartesian()
    CartSlep.gen_Slep_basis(boundary_points, np.double(N), np.array([eval_gridx, eval_gridy]).T)

    # clipping off at the Shannon number
    N2D = None #int(np.sum(CartSlep.V))
    CartSlep.G = CartSlep.G[:,:N2D]
    CartSlep.H = CartSlep.H[:,:N2D]

    # removing the odd basis functions

    # the data we intend to fit to
    vdf_data = np.append(gvdf_tstamp.vdfdata, gvdf_tstamp.vdfdata)

    # performing the inversion
    GTG = CartSlep.G.T @ CartSlep.G
    coeffs = np.linalg.inv(GTG) @ CartSlep.G.T @ vdf_data

    return coeffs, CartSlep

def super_resolution(coeffs, boundary_points, N, plotSlep=False):
    # making the grid on which to evaluate the Slepian functions
    eval_gridx = np.linspace(boundary_points[:,0].min(), boundary_points[:,0].max(), 49)
    # eval_gridx = np.linspace(0.1, boundary_points[:,0].max(), 49)
    eval_gridy = np.linspace(boundary_points[:,1].min(), boundary_points[:,1].max(), 49)
    xx, yy = np.meshgrid(eval_gridx, eval_gridy, indexing='ij')

    CartSlep = eval_Slepians.Slep_2D_Cartesian()
    CartSlep.gen_Slep_basis(boundary_points, np.double(N), np.array([xx.flatten(), yy.flatten()]).T)

    # clipping off at the Shannon number
    N2D = None #int(np.sum(CartSlep.V))

    # constructing the super-resolved grid
    vdfrec = coeffs @ CartSlep.H[:,:N2D].T

    if(plotSlep):
        plotter.plot_CartSlep(xx, yy, CartSlep, tidx)

    return vdfrec, CartSlep, xx, yy