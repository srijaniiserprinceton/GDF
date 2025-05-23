import pickle
import json
import numpy as np

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def credential_reader(cred_file=None):
    if cred_file:
        credentials = load_config(cred_file)
        creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
        return creds
    else:
        return None
    
def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def norm_array(arr):
    arr = np.asarray(arr)
    return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
