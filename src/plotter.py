import numpy as np
import matplotlib.pyplot as plt
import plasmapy.formulary as form
import astropy.constants as c
import astropy.units as u
plt.ion()

def plot_span_vs_rec_scatter(tidx, gvdf, vdf_data, vdf_rec):
    # These are for plotting with the tricontourf routine.
    # getting the plasma frame coordinates
    vpara_pf = gvdf.vpara
    vperp_pf = gvdf.vperp
    vpara_nonan = vpara_pf[tidx, gvdf.nanmask[tidx]]
    vperp_nonan = vperp_pf[tidx, gvdf.nanmask[tidx]]

    vdf_nonan = vdf_data
    vdf_rec_nonan = vdf_rec 
    
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True, layout='constrained')
    ax0 = ax[0].scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=(vdf_nonan), vmin=0, vmax=4)
    ax[0].scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=(vdf_nonan), vmin=0, vmax=4)
    ax[0].set_title('SPAN VDF')
    ax[0].set_ylabel(r'$v_{\parallel}$')
    ax[0].set_xlabel(r'$v_{\perp}$')
    ax[0].set_aspect('equal')
    # making the scatter plot of the gyrotropic VDF
    ax1 = ax[1].scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=0, vmax=4)
    ax[1].scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=0, vmax=4)
    ax[1].set_title('Reconstructed VDF')
    ax[1].set_xlabel(r'$v_{\perp}$')
    ax[1].set_aspect('equal')
    plt.colorbar(ax1)


    plt.show()

def plot_span_vs_rec_contour(gvdf, vdf_data, vdf_rec, tidx=None, GRID=False, VA=None, SAVE=False):
    if VA:
        v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])/VA
        v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])/VA
        xlabel = r'$v_{\perp}/v_{A}$'
        ylabel = r'$v_{\parallel}/v_{A}$'
    else:
        v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
        v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])
        xlabel = r'$v_{\perp}$'
        ylabel = r'$v_{\parallel}$'

    # v_para_all -= gvdf.fac.vshift[tidx]

    vdf_nonan = vdf_data
    
    vdf_data_all = np.concatenate([vdf_nonan, vdf_nonan])
    vdf_rec_all  = np.concatenate([vdf_rec, vdf_rec])
    
    lvls = np.linspace(0, int(np.log10(gvdf.maxval[tidx])+1) - int(np.log10(gvdf.minval[tidx]) - 1), 10)

    zeromask = vdf_rec_all == 0
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True, layout='constrained')
    a0 = ax[0].tricontourf(v_perp_all, v_para_all, vdf_data_all, 
                           cmap='plasma', levels=lvls)#, levels=np.linspace(-23, -19, 10))
    ax[0].set_xlabel(xlabel, fontsize=12)
    ax[0].set_ylabel(ylabel, fontsize=12)
    ax[0].set_aspect('equal')
    ax[0].set_title('SPAN VDF')

    a1 = ax[1].tricontourf(v_perp_all[~zeromask], v_para_all[~zeromask], vdf_rec_all[~zeromask],
                           cmap='plasma', levels=lvls)#, levels=np.linspace(-23, -19, 10))
    ax[1].set_xlabel(xlabel, fontsize=12)
    ax[1].set_aspect('equal')
    ax[1].set_title('Reconstructed VDF')

    plt.colorbar(a1)

    if GRID:
        [ax[i].scatter(v_perp_all[len(v_para_all)//2:,], v_para_all[len(v_para_all)//2:,], color='k', marker='.', s=3) for i in range(2)]

    if SAVE:
        plt.savefig(f'./Figures/span_rec_contour/tricontour_plot_{tidx}')
        plt.close(fig)

    else: plt.show()

def plot_super_resolution(gvdf, tidx, vdf_super, SAVE=False, VDFUNITS=False, VSHIFT=None, DENSITY=None):
    grids = gvdf.grid_points
    mask = gvdf.hull_mask

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')


    if VDFUNITS:
        f_super = np.power(10, vdf_super) * gvdf.minval[tidx]
        lvls = np.linspace(int(np.log10(gvdf.minval[tidx]) - 1), int(np.log10(gvdf.maxval[tidx])+1), 10)
        if VSHIFT:
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0] - VSHIFT, np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0] - VSHIFT, np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf.vshift[tidx], color='k', marker='.', s=3)
            if DENSITY:
                Bmag = np.linalg.norm(gvdf.b_span[tidx])
                VA = form.speeds.Alfven_speed(Bmag * u.nT, DENSITY * u.cm**(-3), ion='p+').to(u.km/u.s)

                ax.arrow(0, 0, 0, VA.value, fc='k', ec='k')


        else:
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.', s=3)
    else:
        ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], vdf_super[mask], levels=np.linspace(0,4.0,10), cmap='plasma')

        ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.', s=3)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(r'$v_{\perp}$', fontsize=19)
    ax.set_ylabel(r'$v_{\parallel}$', fontsize=19)
    ax.set_title(f'Super Resolution | {str(gvdf.l2_time[tidx])[:19]}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_xlim([-400,400])
    ax.set_aspect('equal')

    if SAVE:
        plt.savefig(f'./Figures/super_res/super_resolved_{tidx}_{gvdf.npts}.pdf')
        plt.close(fig)
    else: plt.show()

def plot_gyrospan(gvdf, tidx, vdfdata, SAVE=False, VDFUNITS=False, VSHIFT=None, DENSITY=None):
    v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
    v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])

    vdf_data_all = np.concatenate([vdfdata, vdfdata])    

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')


    if VDFUNITS:
        f_super = np.power(10, vdf_data_all) * gvdf.minval[tidx]
        lvls = np.linspace(int(np.log10(gvdf.minval[tidx]) - 1), int(np.log10(gvdf.maxval[tidx])+1), 10)
        if VSHIFT:
            ax1 = ax.tricontourf(v_perp_all, v_para_all - VSHIFT, np.log10(f_super), levels=lvls, cmap='plasma')
            if DENSITY:
                Bmag = np.linalg.norm(gvdf.b_span[tidx])
                VA = form.speeds.Alfven_speed(Bmag * u.nT, DENSITY * u.cm**(-3), ion='p+').to(u.km/u.s)

                ax.arrow(0, 0, 0, VA.value, fc='k', ec='k')


        else:
            ax1 = ax.tricontourf(v_perp_all, v_para_all - VSHIFT, np.log10(f_super), levels=lvls, cmap='plasma')
    else:
        ax1 = ax.tricontourf(v_perp_all, v_para_all - VSHIFT, vdf_data_all, levels=np.linspace(0,4.0,10), cmap='plasma')
        
    ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf.vshift[tidx], color='k', marker='.', s=3)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(r'$v_{\perp}$', fontsize=19)
    ax.set_ylabel(r'$v_{\parallel}$', fontsize=19)
    ax.set_title(f'Super Resolution | {str(gvdf.l2_time[tidx])[:19]}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim([-400,400])
    ax.set_aspect('equal')
