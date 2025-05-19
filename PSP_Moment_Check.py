import numpy as np
import cdflib
import json
import xarray as xr
import matplotlib.pyplot as plt; plt.ion()
import astropy.constants as c
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical as c2s
NAX = np.newaxis

import src.functions as fn

from scipy.integrate import simpson
from tqdm import tqdm

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # loading VDF and defining timestamp
    trange = ['2020-01-29T00:00:00', '2020-01-29T23:59:59']
    credentials = load_config('./config.json')
    # creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    creds = None
    
    # Initialzise the PSP vdf
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=True)

    # Choose a user defined time index
    # tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-29T18:10:06')))


    dens = {}
    vels = {}

    for tidx in tqdm(range(len(psp_vdf.time.data))):
        vdf    = psp_vdf.vdf.data[tidx]
        energy = psp_vdf.energy.data[tidx]
        theta  = psp_vdf.theta.data[tidx]
        phi    = psp_vdf.phi.data[tidx]

        mass_p = 0.010438870
        sinT = np.sin(np.radians(90.-theta))
        velocity = np.sqrt(2*energy/(mass_p * 1e-10))
        integrand = vdf * sinT * velocity**2

        vx = velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        vy = velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        vz = velocity * np.sin(np.radians(theta)) 

        # Check the density
        n = simpson(simpson(simpson(integrand, x=np.radians(phi), axis=2), x=np.radians(90-theta[:,:,0])), x=velocity[:,0,0])
        dens[tidx] = n

        ux = simpson(simpson(simpson(integrand * vx, x=np.radians(phi), axis=2), x=np.radians(90-theta[:,:,0])), x=velocity[:,0,0])/n
        uy = simpson(simpson(simpson(integrand * vy, x=np.radians(phi), axis=2), x=np.radians(90-theta[:,:,0])), x=velocity[:,0,0])/n
        uz = simpson(simpson(simpson(integrand * vz, x=np.radians(phi), axis=2), x=np.radians(90-theta[:,:,0])), x=velocity[:,0,0])/n

        vels[tidx] = np.array([ux, uy, uz])


    # Load in the SPAN-i moments
    span_moments = fn.init_psp_moms(trange, CREDENTIALS=creds, CLIP=True)


    