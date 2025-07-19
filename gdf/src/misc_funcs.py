import pickle
import json
import numpy as np
import importlib.util

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
