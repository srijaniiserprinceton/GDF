import pickle
import json
import numpy as np
import importlib.util
import xarray as xr

def load_config(file_path):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def credential_reader(cred_file=None):
    """
    Reads the config.json file containing the credentials and returns it as a list.

    Parameters
    ----------
    cred_file : str
        Path to the config.json file.
    """
    if cred_file:
        credentials = load_config(cred_file)
        # TODO: Add FIELDS credentials for variance analysis
        creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
        return creds
    else:
        return None
    
def write_pickle(x, fname):
    """
    Write to a pickle file.

    Parameters
    ----------
    x : data structure
        Contains the dictionary or data structure to be saved into the pickle file.

    fname : str
        Path to the pickle file to be written excluding the .pkl at the end. 
    """
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fname):
    """
    Read from a pickle file.

    Parameters
    ----------
    fname : str
        Path to the pickle file to be read excluding the .pkl at the end. 
    """
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def norm_array(arr):
    """
    Normalizing an input array to range between (0, 1).

    Parameters
    ----------
    arr : array-like of floats
        The array to be normalized.
    """
    arr = np.asarray(arr)
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

def load_config_new(path):
    spec = importlib.util.spec_from_file_location('config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config

def png_to_mp4_converter(image_dir, destination_dir, frames_per_second=10):
    import cv2
    import os
    from natsort import natsorted

    # Parameters
    image_folder = image_dir  # folder with PNGs
    output_video = f'{destination_dir}/output.mp4'
    fps = frames_per_second  # desired frame rate

    # Get image files
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsorted(images)

    # Read first image to get size
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write frames
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

def make_synthetic_cdf(time_array, energy_sort, phi_sort, theta_sort, vdf, bspan, uspan, EFLUX=None, COUNTS=None):
    """
    Script to convert time, energy, phi, theta to a .cdf file matching the format of the SPAN-Ai instrument suite.

    Parameters:
    -----------
    time_array : ndarray
        Corresponding time array value, typically defaults to integer array for subsequent vdf models.
    energy_sort : ndarray 
        Energy array in the shape of (N_E, N_Theta, N_Phi).
    phi_sort : ndarray 
        Phi array in the shape of (N_E, N_Theta, N_Phi). In degrees
    theta_sort : ndarray 
        Theta array in the shape of (N_E, N_Theta, N_Phi). In degrees
    vdf : ndarray
        VDF values evaluated at each energy, theta, and phi grid. 
    bspan : ndarray
        3 x N array corresponding to the model magnetic field for each interval.
    uspan : ndarray
        3 x N array corresponding to the VDF centroid or plasma bulk speed.
    """

    # set the xarray time
    xr_time_array = time_array

    # Zero values.
    if EFLUX == None:
        eflux_sort = np.zeros_like(vdf)
        EFLUX = eflux_sort
    if COUNTS == None:
        count_sort = np.zeros_like(vdf)
        COUNTS = count_sort
    unix_time = np.zeros_like(xr_time_array)

    # Generate the xarray dataArrays for each value we are going to pass
    xr_eflux  = xr.DataArray(eflux_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'eV/cm2-s-ster-eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_energy = xr.DataArray(energy_sort, dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.01', 'validmax' : '100000.', 'scale' : 'log'})
    xr_phi    = xr.DataArray(phi_sort,    dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_theta  = xr.DataArray(theta_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_vdf    = xr.DataArray(vdf,         dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'s^3/cm^6', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_count  = xr.DataArray(count_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'integer', 'fillval' : 'np.array([0], dtype=float32)', 'validmin':'0', 'validmax' : '2048', 'scale' : 'linear'})
    xr_unix   = xr.DataArray(unix_time, dims=['time'], coords=dict(time = xr_time_array), attrs={'units' : 'time', 'description':'Unix time'})
    xr_bspan  = xr.DataArray(bspan, dims=['time', 'coor'], coords=dict(time = xr_time_array, coor=np.arange(3)), attrs={'units' : 'None', 'description':'B-unit vector'})
    xr_uspan   = xr.DataArray(uspan, dims=['time', 'coor'], coords=dict(time = xr_time_array, coor=np.arange(3)), attrs={'units' : 'km/s', 'description':'Approximate bulk speed'})

    # Generate the xarray.Dataset
    xr_ds = xr.Dataset({
                        'unix_time' : xr_unix,
                        'eflux'  : xr_eflux,
                        'energy' : xr_energy,
                        'phi' : xr_phi,
                        'theta' : xr_theta,
                        'vdf' : xr_vdf,
                        'counts' : xr_count, 
                        'b_span': xr_bspan,
                        'u_span': xr_uspan,
                       },
                       attrs={'description' : 'Temp SPAN-i data recast into proper format. VDF unit is in s^3/cm^6.'})
    
    return(xr_ds)

def bvector_rotation_matrix(B_vec):
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

    return np.array([[Nx, Ny, Nz], [Px, Py, Pz], [Qx, Qy, Qz]])



