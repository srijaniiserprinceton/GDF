import numpy as np
from scipy.interpolate import RBFInterpolator as rbf
from numba import njit
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')

import biMax_emcee as fit_biMax

def get_collars(biMax_params, para_absmax=1500, perp_absmax=1000, collar_dist=1000, N=50):
    vpara, vperp = np.mgrid[-para_absmax:0:N+1j, -perp_absmax:perp_absmax:N+1j]

    # only choosing points beyond collar_dist from the centroid of the VDF
    vc, vb, __, __, __, __, __, r = biMax_params
    vpara_centroid = (vc + vb * np.power(10, r)) / (1 + np.power(10, r)) * 100
    dist = np.sqrt(np.square(vpara - vpara_centroid) + np.square(vperp))
    collar_mask = dist >= collar_dist

    collar_vpara, collar_vperp = vpara[collar_mask], vperp[collar_mask]

    # evaluating the biMax VDF on the collar locations
    vdf_collarvals = fit_biMax.biMax_numba(biMax_params, collar_vpara, collar_vperp)

    return vdf_collarvals, collar_vpara, collar_vperp


def perform_tps(vdf_data, vpara_grid, vperp_grid, s=5):
    grid_points = np.stack([vpara_grid, vperp_grid]).T
    finterp = rbf(grid_points, vdf_data, smoothing=s, kernel='thin_plate_spline')

    return finterp

def plot_enhanced_gdf(ftps, para_absmax=2000, perp_absmax=1000, N=100):
    vgrid = np.mgrid[-para_absmax:0:N+1j, -perp_absmax:perp_absmax:N+1j]
    vgrid_flat = vgrid.reshape(2, -1).T
    egdf = ftps(vgrid_flat)
    egdf = np.reshape(egdf, (N, N))

    levels = np.append(np.linspace(-12, -5, 10), np.linspace(-4,0,10))

    # plotting
    plt.figure(figsize=(8,8))
    plt.contourf(vgrid[0], vgrid[1], egdf, cmap='jet', levels=levels)
    plt.colorbar()
    plt.contour(vgrid[0], vgrid[1], egdf, collar_vals, colors='white', ls='--', levels=np.linspace(-70,-30,10))
    plt.scatter(vpara, vperp, marker='x', s=2, c='k')
    plt.gca().set_aspect('equal')
    plt.tight_layout()

if __name__=='__main__':
    tidx = 7300
    # tidx = 19420
    # tidx = 19275

    # reading the GDF file
    gdf = np.load(f'vdf_Sleprec_{tidx}.npy').flatten()
    vpara = np.load(f'vpara_{tidx}.npy').flatten()
    vperp = np.load(f'vperp_{tidx}.npy').flatten()
    baseline = 1e-4

    #-------------------performing the biMax fitting----------------------#
    # dummy biMax function compilation
    dummy_params = np.zeros(8)
    _ = fit_biMax.biMax_numba(dummy_params, np.zeros_like(vpara), np.zeros_like(vperp))

    # initializing the super-resolution class
    supres_vdf = fit_biMax.supres(gdf, vpara, vperp)

    # performing a biMax fitting to obtain the collars
    bimax_best_params, __, __ = supres_vdf.biMaxfit()

    # retreiving the collar values before performing the thin-plate spline interpolation
    collar_vals, collar_vpara, collar_vperp = get_collars(bimax_best_params)

    # concatenating the GDF and bi-Max collar values
    log_gdf = np.log10((gdf / np.nanmax(gdf)) + baseline)
    log_collar = np.log10(collar_vals)
    vdf_data = np.append(log_gdf, log_collar)
    vpara_grid = np.append(vpara, collar_vpara)
    vperp_grid = np.append(vperp, collar_vperp)

    # thin-plate spline interpolation to fill in the gap between observed VDF and collars
    f_interp = perform_tps(vdf_data, vpara_grid, vperp_grid, s=5)

    # plotting the interpolated vdf
    plot_enhanced_gdf(f_interp)