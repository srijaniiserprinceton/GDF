import numpy as np
from scipy.interpolate import RBFInterpolator as rbf
from numba import njit
import matplotlib.pyplot as plt; plt.ion(); plt.style.use('dark_background')

import biMax_emcee as fit_biMax

def get_collars_old(biMax_params, para_absmax=1500, perp_absmax=1000, collar_dist=1000, N=50):
    vpara, vperp = np.mgrid[-600:400:N+1j, -perp_absmax:perp_absmax:N+1j]

    # only choosing points beyond collar_dist from the centroid of the VDF
    vc, vb, __, __, __, __, __, r = biMax_params
    vpara_centroid = (vc + vb * np.power(10, r)) / (1 + np.power(10, r)) * 100
    dist = np.sqrt(np.square(vpara - vpara_centroid) + np.square(vperp))
    collar_mask = dist >= collar_dist

    collar_vpara, collar_vperp = vpara[collar_mask], vperp[collar_mask]

    # evaluating the biMax VDF on the collar locations
    vdf_collarvals = fit_biMax.biMax_numba(biMax_params, collar_vpara, collar_vperp)

    return vdf_collarvals, collar_vpara, collar_vperp

def get_collars(biMax_params, para_absmax=1500, perp_absmax=1000, collar_dist=1000, N=50, threshold=1e-6):
    vpara, vperp = np.mgrid[-600:0:N+1j, -perp_absmax:perp_absmax:N+1j]

    # evaluating the biMax VDF on the collar locations
    vdf_collarvals = fit_biMax.biMax_numba(biMax_params, vpara, vperp)

    mask = vdf_collarvals < vdf_collarvals.max() * threshold

    collar_vpara = vpara[mask]
    collar_vperp = vperp[mask]
    vdf_collarvals = vdf_collarvals[mask]


    return vdf_collarvals, collar_vpara, collar_vperp


def perform_tps(vdf_data, vpara_grid, vperp_grid, s=5):
    grid_points = np.stack([vpara_grid, vperp_grid]).T
    finterp = rbf(grid_points, vdf_data, smoothing=s, kernel='thin_plate_spline')

    return finterp

def plot_enhanced_gdf(ftps, para_absmax=2000, perp_absmax=1000, N=100):
    vgrid = np.mgrid[-1000:1000:N+1j, -perp_absmax:perp_absmax:N+1j]
    vgrid_flat = vgrid.reshape(2, -1).T
    egdf = ftps(vgrid_flat)
    egdf = np.reshape(egdf, (N, N))

    levels = np.append(np.linspace(-12, -5, 10), np.linspace(-4,0,10))

    # plotting
    plt.figure(figsize=(8,8))
    plt.contourf(vgrid[0], vgrid[1], egdf, cmap='jet', levels=levels)
    plt.colorbar()
    plt.scatter(collar_vpara, collar_vperp, c=collar_vals, s=1) #colors='white', ls='--', levels=np.linspace(-70,-30,10))
    plt.scatter(vpara, vperp, marker='x', s=2, c='k')
    plt.gca().set_aspect('equal')
    plt.tight_layout()

if __name__=='__main__':
    tidx = 7290
    # tidx = 19420
    # tidx = 19275

    # reading the GDF file
    # gdf = np.load(f'Slepian_Rec_VDFs/vdf_Sleprec_2020-01-26T14:10:38.npy').flatten()
    # vpara = np.load(f'Slepian_Rec_VDFs/vpara_2020-01-26T14:10:38.npy').flatten()
    # vperp = np.load(f'Slepian_Rec_VDFs/vperp_2020-01-26T14:10:38.npy').flatten()
    gdf = np.load('vdf_Sleprec_7300.npy').flatten()
    vpara = np.load('vpara_7300.npy').flatten()
    vperp = np.load('vperp_7300.npy').flatten()
    baseline = 1e-4

    gdf[gdf == np.nanmin(gdf)] = np.nan

    mask = np.isnan(gdf)
    gdf = gdf[~mask]
    vpara = vpara[~mask]
    vperp = vperp[~mask]

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

    plt.figure()
    plt.tricontourf(collar_vpara, collar_vperp, collar_vals)

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


    fig, ax = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True, layout='constrained')
    # levels = np.linspace(8, 11, 10) - 30 
    axs = ax[0].tricontourf(vpara, vperp, log_gdf, cmap='plasma', levels=np.linspace(-4,0,10))
    # ax[0].contour(VX, VY, np.log10(fit), cmap='jet')
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(axs)
    
    vgrid = np.mgrid[-600:400:101j, -500:500:101j]
    vgrid_flat = vgrid.reshape(2, -1).T
    egdf = f_interp(vgrid_flat)
    egdf = np.reshape(egdf, (101,101))


    axs1 = ax[1].contourf(vgrid[0], vgrid[1], egdf, cmap='plasma', levels=np.linspace(-4,0,10))

    # fig.colorbar(ax=axs)
    fig.colorbar(axs1)