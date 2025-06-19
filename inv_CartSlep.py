import sys, importlib
import numpy as np
from line_profiler import profile
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, LineString

from gdf.src import hybrid_ubulk
from gdf.src import functions as fn
from gdf.src import misc_funcs as misc_fn
from gdf.src import eval_Slepians
from gdf.src import plotter

@profile
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

@profile
def super_resolution(coeffs, boundary_points, N, plotSlep=False):
    # making the grid on which to evaluate the Slepian functions
    eval_gridx = np.linspace(boundary_points[:,0].min(), boundary_points[:,0].max(), 49)
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

@profile
def do_Voronoi(points):
    # Constructing the Voronoi cells for the set of points
    vor = Voronoi(points)
    # Constructing the convex hull around the points
    hull = ConvexHull(points)
    hull_polygon = Polygon(points[hull.vertices])

    def polygon_area(points):
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

    # Get the vertices of the convex hull
    hull_vertices = points[hull.vertices]

    # Calculate the area of the convex hull
    total_area = polygon_area(hull_vertices)

    # Prepare output
    clipped_regions = []
    clipped_areas = []
    valid_points = []

    for i, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue  # skip infinite or degenerate
        poly = Polygon(vor.vertices[region])
        
        # Clip polygon with convex hull
        clipped = poly.intersection(hull_polygon)
        if not clipped.is_empty and clipped.area > 0:
            clipped_regions.append(clipped)
            clipped_areas.append(clipped.area)
            valid_points.append(points[i])

    # Find max area
    clipped_areas = np.array(clipped_areas)
    max_index = np.argsort(clipped_areas)
    valid_points = np.array(valid_points)

    # sorting the points by areas
    sortidx = np.argsort(clipped_areas)[::-1]

    return total_area, clipped_areas, sortidx, valid_points

def get_Nbasis(Atotal, Acells, P, VC, nlargest=3):
    K = (np.pi - 15*np.pi/180)/VC * (P[:,0] / np.sqrt(2*Acells))
    N = K**2 * Atotal / (4 * np.pi)

    return np.min([np.mean(N[:nlargest*2]), 14])

if __name__=='__main__':
    # importing the config file provided at command line
    config_file = sys.argv[1]
    config = importlib.import_module(config_file)
    gvdf_tstamp = hybrid_ubulk.run(config)
    tidx = config.START_INDEX

    # points for Voronoi tessellation
    Voronoi_points = np.vstack([gvdf_tstamp.v_para_all, gvdf_tstamp.v_perp_all]).T
    total_area, clipped_areas, sortidx, valid_points = do_Voronoi(Voronoi_points)

    # the Shannon number for generating the 2D Slepians
    # Note that the maximum horizontal wavenumber, k = sqrt(4 * pi * N / A)
    N = int(get_Nbasis(total_area, clipped_areas[sortidx], valid_points[sortidx],
            gvdf_tstamp.vshift, nlargest=3))

    # the SPAN-i data to be used
    vdf_data = np.append(gvdf_tstamp.vdfdata, gvdf_tstamp.vdfdata)

    # getting the coefficients for the Slepian basis
    coeffs, CartSlep_lr = inversion_CartSlep(gvdf_tstamp, N)
    # constructing the high-res Slepians and generating the final VDF
    vdfrec, CartSlep_hr, xx, yy = super_resolution(coeffs, CartSlep_lr.XY, N, plotSlep=True)

    # converting the VDFs to SPAN-i consistent units
    f_supres = np.power(10, vdfrec) * gvdf_tstamp.minval[tidx]
    f_data = np.power(10, vdf_data) * gvdf_tstamp.minval[tidx]

    # plotting the high resolution basis functions
    plotter.plot_CartSlep(xx, yy, CartSlep_hr, tidx)
    # plotting the final reconstruction
    plotter.plot_supres_CartSlep(gvdf_tstamp, CartSlep_hr, xx, yy, f_data, f_supres, tidx)