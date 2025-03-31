import numpy as np
import scipy 
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()


def biMax_dist(vpara, vperp, n=5, upara=0, wperp=50, wpara=50):
    # This is a bi-Maxwellian in the plasma frame.
    upara *= 100
    wperp *= 100
    wpara *= 100 
    return np.power(10,n)/(np.pi**(3/2) * wperp * wpara) * np.exp(-vperp**2/wperp**2 - (vpara - upara)**2/wpara**2)

def biMax_model(vpara, vperp, fit_params):
    ncore, ucore, w1core, w2core, nbeam, ubeam, w1beam, w2beam = fit_params

    corefit = biMax_dist(vpara, vperp, ncore, ucore, w1core, w2core)
    beamfit = biMax_dist(vpara, vperp, nbeam, ubeam, w1beam, w2beam)

    return(corefit + beamfit)

def residuals(fit_params, vpara, vperp, vdf):
    fit = biMax_model(vpara, vperp, fit_params)
    return(abs(vdf)*np.abs(np.log10(vdf) - np.log10(fit))**2)
    # return(np.abs(np.log10(vdf) - np.log10(fit)**2))


if __name__ == "__main__":
    vdf_rec = np.load('vdf_Sleprec.npy').flatten()
    # vdf_rec = vdf_rec
    # vdf_rec[np.log10(vdf_rec) < -25] = np.nan
    vdf_rec = vdf_rec
    vpara = np.load('vpara.npy').flatten()
    vperp = np.load('vperp.npy').flatten()

    # Mask out nan values
    mask = np.isnan(vdf_rec)
    vdf_rec = vdf_rec[~mask]
    vpara   = vpara[~mask]
    vperp   = vperp[~mask]

    guesses = [8, -4, .5, .5, 8, -6, .4, .4] + np.random.rand(1)/5
    
    fitout = opt.leastsq(residuals, guesses, args=(vpara, vperp, vdf_rec), full_output=True)

    vparafit = np.linspace(np.nanmax(vpara), np.nanmin(vpara))
    vperpfit = np.linspace(-np.nanmax(vperp), np.nanmax(vperp))

    VX, VY = np.meshgrid(vparafit, vperpfit, indexing='ij')

    fit = biMax_model(VX, VY, fitout[0])


    fig, ax = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True, layout='constrained')
    levels = np.linspace(8, 11, 10) - 30 
    axs = ax[0].tricontourf(vpara, vperp, np.log10(vdf_rec), cmap='plasma', levels=levels)
    ax[0].contour(VX, VY, np.log10(fit), cmap='jet')

    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(axs)

    axs1 = ax[1].contourf(VX, VY, np.log10(fit), cmap='plasma', levels=levels)

    # fig.colorbar(ax=axs)
    fig.colorbar(axs1)
    