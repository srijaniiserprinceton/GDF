import os
import sys, json
import numpy as np
from numba import njit
import xarray as xr
import pyspedas
import cdflib
import glob
from scipy.integrate import simpson as simps
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt; plt.ion(); plt.rcParams['font.size'] = 16

from datetime import datetime
from pathlib import Path

"""
Update: Fixed the init routines.
"""

def read_config():
    package_dir = os.getcwd()  
    with open(f"{package_dir}/.config", "r") as f:
        dirnames = f.read().splitlines()

    return dirnames

def get_latest_version(file_names):
    latest_version = -1
    latest_file = None
    
    for file_name in file_names:
        # Extract the version number as an integer
        version_str = file_name.split('_v')[-1].split('.')[0]
        version = int(version_str)
        
        # Update the latest version and file name if current version is higher
        if version > latest_version:
            latest_version = version
            latest_file = file_name
    
    return latest_file

def _get_psp_vdf(trange, CREDENTIALS=None, OVERRIDE=False):
    '''
    Get and download the latest version of PSP data. 

    Parameters:
    -----------
    trange : list of str, datetime object
             Timerange to download the data
    probe : int or list of ints
            Which MMS probe to get the data from.
    
    Returns:
    --------

    TODO : Add check if file is already downloaded and use local file.
    TODO : Replace with a cdaweb or wget download procedure.
    '''
    date = datetime.strptime(trange[0], '%Y-%m-%dT%H:%M:%S')
    date_string = date.strftime('%Y%m%d')
    
    # Get all the key information
    pwd = os.getcwd()

    files = None
    if (os.path.exists(f'{pwd}/psp_data/sweap/spi')) and (OVERRIDE == False):
        preamble = 'psp_swp_spi_sf00'
        if CREDENTIALS:
            # Credentials means we are using the private data directory.
            level = 'L2'
            dtype = '8Dx32Ex8A'

            file = f'{os.getcwd()}/psp_data/sweap/spi/{level}/spi_sf00/{date.year}/{date.month:02d}/{preamble}_{level}_{dtype}_{date_string}_v**.cdf'
        else:
            # Loading in the public side of the data.
            level = 'l2'
            dtype = '8dx32ex8a'

            file = f'{os.getcwd()}/psp_data/sweap/spi/{level}/spi_sf00_{dtype}/{date.year}/{preamble}_{level}_{dtype}_{date_string}_v**.cdf'

        if (glob.glob(file)):
            print('Data is already downloaded', flush = True)
            latest_version = get_latest_version(glob.glob(file))

            files = [latest_version]

        if files == None:
            if CREDENTIALS:
                files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L2', notplot=True, time_clip=True, downloadonly=True, last_version=True, get_support_data=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
            else:
                files = pyspedas.psp.spi(trange, datatype='spi_sf00_8dx32ex8a', level='l2', notplot=True, time_clip=True, downloadonly=True, last_version=True, get_support_data=True)

            

    else:
        if CREDENTIALS:
            files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L2', notplot=True, time_clip=True, downloadonly=True, last_version=True, get_support_data=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
        else:
            files = pyspedas.psp.spi(trange, datatype='spi_sf00_8dx32ex8a', level='l2', notplot=True, time_clip=True, downloadonly=True, last_version=True, get_support_data=True)

    return(files)

def init_psp_vdf(trange, CREDENTIALS=None, CLIP=False, filename=None):
    '''
    Parameters:
    -----------
    filename : list containing the files that are going to be loaded in.

    Returns:
    --------
    vdf_ds : xarray dataset containing the key VDF parameters from the given filename.
    
    NOTE: This will only load in a single day of data.
    '''
    # Constants
    mass_p = 0.010438870        # eV/(km^2/s^2)
    charge_p = 1

    if filename:
        files = [filename]
    else:
        files = _get_psp_vdf(trange, CREDENTIALS)

    if len(files) > 1:
        xr_data = xr.concat([cdflib.cdf_to_xarray(f).drop_vars(['ROTMAT_SC_INST']) for f in files], dim='Epoch')
    else:
        xr_data = cdflib.cdf_to_xarray(*files)

    # Get the instrument time
    xr_time_object = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(xr_data.Epoch.data)
    xr_time_array  = xr_time_object.utc.datetime    # Ensure we are in utc!

    # Keep the unix time as a check
    unix_time = xr_data.TIME.data

    # Now swap xr_data.Epoch to be in terms of time
    xr_data['Epoch'] = xr_time_array
    # Clip the dataset if CLIP flag is set to be true.
    if CLIP is True:
        xr_data = xr_data.sel(Epoch=slice(trange[0], trange[-1]))

        xr_time_array = xr_data.Epoch.data
        unix_time = xr_data.TIME.data

    # Differential energy flux taken from PSP
    energy_flux = xr_data.EFLUX.data

    energy = xr_data.ENERGY.data
    theta  = xr_data.THETA.data
    phi    = xr_data.PHI.data

    counts = xr_data.DATA.data

    theta_dim = 8
    phi_dim = 8
    energy_dim = 32

    LEN = energy_flux.shape[0]

    # Now reshape all of our data: phi_dim, energy_dim, phi_dim
    eflux_sort  = energy_flux.reshape(LEN, phi_dim, energy_dim, theta_dim)
    theta_sort  = theta.reshape(LEN, phi_dim, energy_dim, theta_dim)
    phi_sort    = phi.reshape(LEN, phi_dim, energy_dim, theta_dim)
    energy_sort = energy.reshape(LEN, phi_dim, energy_dim, theta_dim)

    count_sort  = counts.reshape(LEN, phi_dim, energy_dim, theta_dim)

    # Convert the data to be in uniform shape (E, theta, phi)
    eflux_sort  = np.transpose(eflux_sort, [0, 2, 3, 1])
    theta_sort  = np.transpose(theta_sort, [0, 2, 3, 1])
    phi_sort    = np.transpose(phi_sort, [0, 2, 3, 1])
    energy_sort = np.transpose(energy_sort, [0, 2, 3, 1])
    count_sort  = np.transpose(count_sort, [0, 2, 3, 1])

    # Resort the arrays so the energy is increasing
    eflux_sort  = eflux_sort[:, ::-1, :, :]  
    theta_sort  = theta_sort[:, ::-1, :, :]  
    phi_sort    = phi_sort[:, ::-1, :, :]    
    energy_sort = energy_sort[:, ::-1, :, :]
    count_sort  = count_sort[:, ::-1, :, :]

    # Convert energy flux into differential energy flux
    vdf = eflux_sort * ((mass_p * 1e-10)**2) /(2 * energy_sort**2)      # 1e-10 is used to convert km^2 to cm^2

    # number_flux = eflux_sort/energy_sort
    # vdf = number_flux * (mass_p**2)/((2E-5)*energy_sort)

    # Generate the xarray dataArrays for each value we are going to pass
    xr_eflux  = xr.DataArray(eflux_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'eV/cm2-s-ster-eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_energy = xr.DataArray(energy_sort, dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.01', 'validmax' : '100000.', 'scale' : 'log'})
    xr_phi    = xr.DataArray(phi_sort,    dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_theta  = xr.DataArray(theta_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_vdf    = xr.DataArray(vdf,         dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'s^3/cm^6', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_count  = xr.DataArray(count_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'integer', 'fillval' : 'np.array([0], dtype=float32)', 'validmin':'0', 'validmax' : '2048', 'scale' : 'linear'})
    xr_unix   = xr.DataArray(unix_time, dims=['time'], coords=dict(time = xr_time_array), attrs={'units' : 'time', 'description':'Unix time'}) 

    # Generate the xarray.Dataset
    xr_ds = xr.Dataset({
                        'unix_time' : xr_unix,
                        'eflux'  : xr_eflux,
                        'energy' : xr_energy,
                        'phi' : xr_phi,
                        'theta' : xr_theta,
                        'vdf' : xr_vdf,
                        'counts' : xr_count
                       },
                       attrs={'description' : 'SPAN-i data recast into proper format. VDF unit is in s^3/cm^6.'})
    
    return(xr_ds)

def _get_psp_span_mom(trange, CREDENTIALS=None, OVERRIDE=False):
    '''
    Get and download the latest version of the MMS data. 

    Parameters:
    -----------
    trange : list of str, datetime object
             Timerange to download the data
    probe : int or list of ints
            Which MMS probe to get the data from.
    
    Returns:
    --------

    TODO : Add check if file is already downloaded and use local file.
    TODO : Replace with a cdaweb or wget download procedure.
    '''
    date = datetime.strptime(trange[0], '%Y-%m-%dT%H:%M:%S')
    date_string = date.strftime('%Y%m%d')
    
    # Get all the key information
    pwd = os.getcwd()

    files = None
    if (os.path.exists(f'{pwd}/psp_data/sweap/spi/')) and (OVERRIDE == False):
        preamble = 'psp_swp_spi_sf00'
        if CREDENTIALS:
            # Credentials means we are using the private data directory.
            level = 'L3'
            dtype = 'mom'

            file = f'{os.getcwd()}/psp_data/sweap/spi/{level}/spi_sf00/{date.year}/{date.month:02d}/{preamble}_{level}_{dtype}_{date_string}_v**.cdf'
        else:
            # Loading in the public side of the data.
            level = 'l3'
            dtype = 'mom'

            file = f'{os.getcwd()}/psp_data/sweap/spi/{level}/spi_sf00_{level}_{dtype}/{date.year}/{preamble}_{level}_{dtype}_{date_string}_v**.cdf'

        if (glob.glob(file)):
            print('Data is already downloaded', flush = True)
            latest_version = get_latest_version(glob.glob(file))

            files = [latest_version]

        if files == None:
            if CREDENTIALS:
                files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L3', notplot=True, time_clip=True, downloadonly=True, last_version=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
            else:
                files = pyspedas.psp.spi(trange, datatype='spi_sf00_l3_mom', level='l3', notplot=True, time_clip=True, downloadonly=True, last_version=True)


    else:
        if CREDENTIALS:
            files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L3', notplot=True, time_clip=True, downloadonly=True, last_version=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
        else:
            files = pyspedas.psp.spi(trange, datatype='spi_sf00_l3_mom', level='l3', notplot=True, time_clip=True, downloadonly=True, last_version=True)

    return(files)

def init_psp_moms(trange, CREDENTIALS=None, CLIP=False):
    files = _get_psp_span_mom(trange, CREDENTIALS=CREDENTIALS)
    # Check if there are multiple datasets loaded for the interval.
    if len(files) > 1:
        xr_data = xr.concat([cdflib.cdf_to_xarray(f).drop_vars(['ROTMAT_SC_INST']) for f in files], dim='Epoch')
    else:
        xr_data = cdflib.cdf_to_xarray(*files)

    xr_time_object = cdflib.epochs_astropy.CDFAstropy.convert_to_astropy(xr_data.Epoch.data)
    xr_time_array = xr_time_object.utc.datetime 

    xr_data['Epoch'] = xr_time_array
    if CLIP is True:
        xr_data = xr_data.sel(Epoch=slice(trange[0], trange[-1]))

        xr_time_array = xr_data.Epoch.data
    
    return(xr_data)

def field_aligned_coordinates(B_vec):
    if B_vec.shape[0] > 3:
        Bmag = np.nanmean(np.linalg.norm(B_vec, axis=1))

        # The defined unit vector
        Nx = B_vec[:,0]/Bmag
        Ny = B_vec[:,1]/Bmag
        Nz = B_vec[:,2]/Bmag

        # Some random unit vector
        Rx = np.zeros(len(Nx))
        Ry = np.ones(len(Ny))
        Rz = np.zeros(len(Nz))

        # Get the first perp component
        TEMP_Px = (Ny * Rz) - (Nz * Ry)
        TEMP_Py = (Nz * Rx) - (Nx * Rz)
        TEMP_Pz = (Nx * Ry) - (Ny * Rx)

        Pmag = np.sqrt(TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2)

        Px = TEMP_Px / Pmag
        Py = TEMP_Py / Pmag
        Pz = TEMP_Pz / Pmag

        Qx = (Pz * Ny) - (Py * Nz)
        Qy = (Px * Nz) - (Pz * Nx)
        Qz = (Py * Nx) - (Px * Ny)

        return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)
    else:
        Bmag = np.linalg.norm(B_vec)

        # The defined unit vector
        Nx = B_vec[0]/Bmag
        Ny = B_vec[1]/Bmag
        Nz = B_vec[2]/Bmag

        # Some random unit vector
        Rx = 0
        Ry = 1
        Rz = 0

        # Get the first perp component
        TEMP_Px = (Ny * Rz) - (Nz * Ry)
        TEMP_Py = (Nz * Rx) - (Nx * Rz)
        TEMP_Pz = (Nx * Ry) - (Ny * Rx)

        Pmag = np.sqrt(TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2)

        Px = TEMP_Px / Pmag
        Py = TEMP_Py / Pmag
        Pz = TEMP_Pz / Pmag

        Qx = (Pz * Ny) - (Py * Nz)
        Qy = (Px * Nz) - (Pz * Nx)
        Qz = (Py * Nx) - (Px * Ny)

        return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)

def rotate_vector_field_aligned(Ax, Ay, Az, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    # For some Vector A in the SAME COORDINATE SYSTEM AS THE ORIGINAL B-FIELD VECTOR:
    if Ax.ndim == 4:
        An = (Ax * Nx[:, None, None, None]) + (Ay * Ny[:, None, None, None]) + (Az * Nz[:, None, None, None])  # A dot N = A_parallel
        Ap = (Ax * Px[:, None, None, None]) + (Ay * Py[:, None, None, None]) + (Az * Pz[:, None, None, None])  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
        Aq = (Ax * Qx[:, None, None, None]) + (Ay * Qy[:, None, None, None]) + (Az * Qz[:, None, None, None])  # 
    
    else:
        An = (Ax * Nx) + (Ay * Ny) + (Az * Nz)  # A dot N = A_parallel
        Ap = (Ax * Px) + (Ay * Py) + (Az * Pz)  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
        Aq = (Ax * Qx) + (Ay * Qy) + (Az * Qz)  # 

    return(An, Ap, Aq)

def inverse_rotate_vector_field_aligned(Ax, Ay, Az, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    if Ax.ndim == 4:
        An = (Ax * Nx[:, None, None, None]) + (Ay * Px[:, None, None, None]) + (Az * Qx[:, None, None, None])  # A dot N = A_parallel
        Ap = (Ax * Ny[:, None, None, None]) + (Ay * Py[:, None, None, None]) + (Az * Qy[:, None, None, None])  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
        Aq = (Ax * Nz[:, None, None, None]) + (Ay * Pz[:, None, None, None]) + (Az * Qz[:, None, None, None])  # 
    
    else:
        An = (Ax * Nx) + (Ay * Px) + (Az * Qx)  # A dot N = A_parallel
        Ap = (Ax * Ny) + (Ay * Py) + (Az * Qy)  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
        Aq = (Ax * Nz) + (Ay * Pz) + (Az * Qz)  # 

    return(An, Ap, Aq)


# TODO: MAKE ARBITRARY VDF MOMENT CALCULATOR
def vdf_moments(gvdf, vdf_super, tidx):
    # for hybrid when vdf_super is a tuple, choosing the cartesian super-resolution
    if(isinstance(vdf_super, tuple)):
        vdf_super = vdf_super[0]
        # transposing Bf for grid compatibility
        vdfT = np.reshape(vdf_super, (gvdf.nptsx, gvdf.nptsy))
        
        vdf_super = np.transpose(vdfT).flatten()

    minval = gvdf.minval[tidx]
    maxval = gvdf.maxval[tidx]
    grids = gvdf.grid_points
    vx = np.reshape(grids[:,0], (gvdf.nptsx, gvdf.nptsy))
    vy = np.reshape(grids[:,1], (gvdf.nptsx, gvdf.nptsy))
    dx = vx[0,1] - vx[0,0]
    dy = vy[1,0] - vy[0,0]

    mask = gvdf.hull_mask
    mask2 = grids[mask,0] >= 0

    f_super = np.power(10, vdf_super) * minval

    f_super[f_super > 5*maxval] = 0.0

    density = 2*np.pi*np.sum(grids[mask,0][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5)
    velocity = (2*np.pi*np.sum(grids[mask,1][mask2] * 1e5 * grids[mask,0][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5))

    vpara = (velocity/density)

    m_p = 1.6726e-24        # g        
    k_b = 1.380649e-16      # erg/K

    T_para = (m_p/k_b)*(2*np.pi*np.sum((grids[mask,1][mask2] * 1e5 - vpara)**2 * grids[mask,0][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5)/density)
    T_perp = (m_p/(2*k_b))*(2*np.pi*np.sum((grids[mask,0][mask2] * 1e5)**2 * grids[mask,0][mask2]*1e5 * f_super[mask][mask2] * dx*1e5 * dy*1e5)/density)

    T_comp = T_para, T_perp
    T_trace = (T_para + 2*T_perp)/3

    return(density, vpara/1e5, T_comp, T_trace)

def norm_eval_theta(S1, S2, theta=np.linspace(0,180,360)):
    r"""
    Function to evaluate the inner product norm of two functions :math:`S_1(\theta)` and 
    :math:`S_2(\theta)` on a custom grid of :math:`\theta`.

    Parameters
    ----------
    S1 : array-like
        Array containing the values of function :math:`S_1(\theta_i)`.
    S2 : array-like
        Array containing the values of function :math:`S_2(\theta_i)`.
    theta : array-like
        Array containing the values of the :math:`\theta` grid in degrees.

    Returns
    -------
    normval : float
        The scalar normalization value of :math:`\mathcal{N}_{\alpha} = \int S_{\alpha}(\theta) \, S_{\alpha}(\theta) \, \sin(\theta) \, d\theta`.
    """
    normval = simps(S1 * S2 * np.sin(np.radians(theta)), x=np.radians(theta))
    return normval

# testing out different L-curve knee detection algorithms
def compute_lcurve_corner(norms, residuals):
    """
    Find the 'corner' of the L-curve (max curvature) in log-log space.
    Returns the index of the optimal lambda.
    """
    x = norms
    y = residuals

    # Compute first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Compute curvature κ using parametric form
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5

    # Find the index of max curvature
    knee_index = np.argmax(curvature)
    return knee_index

def geometric_knee(x, y):
    """
    Finding the optimal point using a geometric knee detection algorithm of a trade-off curve.
    x and y are the respective axes values in the trade-off curve. It is expected one of them
    is the model-misfit and the other is the data-misfit.

    Returns
    -------
    Returns the index corresponding to the knee of the 1D L-curve.

    """
    x, y = np.array(x), np.array(y)
    # Line: from first to last point
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    point_vecs = np.stack([x - x[0], y - y[0]], axis=1)
    proj_lens = point_vecs @ line_vec_norm
    proj_points = np.outer(proj_lens, line_vec_norm) + np.array([x[0], y[0]])
    distances = np.linalg.norm(point_vecs - (proj_points - np.array([x[0], y[0]])), axis=1)
    return np.argmax(distances)

def merge_bins(bin_edges, counts, threshold):
    """
    Function to merge bins in log10 space for knots until each bin has atleast the desired
    threshold of counts.

    Parameters
    ----------
    bin_edges : array-like of float
        The bin edges of length (len(hist)+1).

    counts : array-like of int
        The histogram count per bin.

    threshold : int
        The minimum count needed per bin. If a bin has lower than this count, then its merged.

    Returns
    -------
    merged_edges : array-like of float
        The 1D array of bin edges after being merged to have counts at a minimum of threshold.

    merged_counts : array-like of int
        The 1D array of counts per merged bins. Still in log10 space of knots locations.
    """
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

def find_supres_grid_and_boundary(xpoints, ypoints, NPTS, plothull=False):
    """
    Parameters
    ----------
    xpoints : array-like of floats
        vperp from the SPAN-i grids with counts above threshold, symmetrized about the vpara axis.

    ypoints : array-like of floats
        vpara from the SPAN-i grids with counts above threshold, symmetrized about the vpara axis. 

    NPTS : tuple of ints
        Tuple of grids (NPTSx, NPTSy) which is the shape of the final super-resolved grid.

    plothull : bool (optional)
        Optional flag to enable plotting the convex hull along with the points, for debugging purposes.

    Returns
    -------
    y : array-like of floats
        1D array containing the regularly spaced grid in vpara.

    supres_grids : array-like of floats
        2D array of shape (NPTSx x NPTSy, 2) containing all the vperp and vpara points (note the 
        first axis is flattened).

    boundary_points : array-like of floats
        This is a (Nboundary_points, 2) shape array containing the coordinates of the boundary grid points.

    hull_mask : array-like of bool
        This is a (NPTSx, NPTSy) boolean array which serves as a mask when computing moments to ensure
        that the calculations are only restricted to within the convex hull.
    """
    points = np.vstack([ypoints, xpoints]).T
    tri = Delaunay(points)
    idx = np.unique(tri.convex_hull)
    points_idx = np.flip(points[idx], axis=1)
    angles = np.arctan2(points_idx[:,0], points_idx[:,1] - np.mean(points_idx[:,1]))
    sortidx = np.argsort(angles)
    points_sorted = points_idx[sortidx]
    boundary_points = np.vstack([points_sorted, points_sorted[0]])

    if(plothull):
        plt.figure()
        plt.scatter(xpoints, ypoints, color='k')
        plt.plot(boundary_points[:,0], boundary_points[:,1], '--r')
        plt.gca().set_aspect('equal')
        plt.xlabel('X', fontweight='bold', fontsize=14)
        plt.ylabel('Y', fontweight='bold', fontsize=14)
        plt.tight_layout()

    # finding the grid for super resolution
    x = np.linspace(boundary_points[:,0].min(), boundary_points[:,0].max(), NPTS[0])
    y = np.linspace(boundary_points[:,1].min(), boundary_points[:,1].max(), NPTS[1])

    # find_simplex seems to like this convention (only to find the hull_mask)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    supres_grids = np.vstack([yy.flatten(), xx.flatten()]).T
    hull_mask = tri.find_simplex(supres_grids) >= 0    # a Mask for the points inside the domain!

    # making the shapes compatible with the convention
    xx, yy = np.meshgrid(x, y, indexing='xy')
    supres_grids = np.vstack([xx.flatten(), yy.flatten()]).T

    # returned points can be simply plotted to give a closed contour circumscribing all points
    return y, supres_grids, boundary_points, hull_mask