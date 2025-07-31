import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plasmapy.formulary as form
import astropy.constants as c
import astropy.units as u
from matplotlib import ticker
plt.rcParams['font.size'] = 16
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
    ax[0].set_ylabel(r'$v_{\parallel}$ [km/s]')
    ax[0].set_xlabel(r'$v_{\perp}$ [km/s]')
    ax[0].set_aspect('equal')
    # making the scatter plot of the gyrotropic VDF
    ax1 = ax[1].scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=0, vmax=4)
    ax[1].scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=vdf_rec_nonan, vmin=0, vmax=4)
    ax[1].set_title('Reconstructed VDF')
    ax[1].set_xlabel(r'$v_{\perp}$ [km/s]')
    ax[1].set_aspect('equal')
    plt.colorbar(ax1)


    plt.show()

def plot_span_vs_rec_contour_POLCAP(gvdf, vdf_data, vdf_rec, tidx, GRID=False, VA=None, SAVE=False, ext='png'):
    if VA:
        v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])/VA
        v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])/VA
        xlabel = r'$v_{\perp}/v_{A}$'
        ylabel = r'$v_{\parallel}/v_{A}$'
    else:
        v_para_all = np.concatenate([gvdf.vpara_nonan, gvdf.vpara_nonan])
        v_perp_all = np.concatenate([-gvdf.vperp_nonan, gvdf.vperp_nonan])
        xlabel = r'$v_{\perp}$ [km/s]'
        ylabel = r'$v_{\parallel}$ [km/s]'

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
        plt.savefig(f'./Figures/span_rec_polcap/tricontour_plot_{tidx}.{ext}')
        plt.close(fig)

    else: plt.show()

def plot_super_resolution_POLCAP(gvdf, vdf_super, mu, tidx, SAVE=False, VDFUNITS=False, VSHIFT=None, DENSITY=None, ext='png'):
    grids = gvdf.grid_points
    mask = gvdf.hull_mask

    fig, ax = plt.subplots(figsize=(8,6), layout='constrained')


    if VDFUNITS:
        f_super = np.power(10, vdf_super) * gvdf.minval[tidx]
        f_data = np.power(10, gvdf.vdfdata) * gvdf.minval[tidx]
        lvls = np.linspace(int(np.log10(gvdf.minval[tidx]) - 1), int(np.log10(gvdf.maxval[tidx])+1), 10)
        cmap = plt.cm.plasma
        norm = colors.BoundaryNorm(lvls, cmap.N)
        if VSHIFT:
            ax1 = ax.tricontourf(grids[mask,0], grids[mask,1] - VSHIFT, np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            # ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf.vshift[tidx], color='k', marker='.', s=3)
            # ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - VSHIFT, color='k', marker='.', s=3)
            plt.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - VSHIFT, c=np.log10(f_data), s=50, cmap='plasma', norm=norm)
            # plt.scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan - VSHIFT, c=np.log10(f_data), s=50, cmap='plasma', norm=norm)
            if DENSITY:
                Bmag = np.linalg.norm(gvdf.b_span[tidx])
                VA = form.speeds.Alfven_speed(Bmag * u.nT, DENSITY * u.cm**(-3), ion='p+').to(u.km/u.s)

                ax.arrow(0, 0, 0, VA.value, fc='k', ec='k')


        else:
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], np.log10(f_super[mask]), levels=lvls, cmap='plasma')
            # ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.', s=3)
            plt.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=np.log10(f_data), s=50, cmap='plasma', norm=norm)
            # plt.scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=np.log10(f_data), s=50, cmap='plasma', norm=norm)
    else:
        ax1 = ax.tricontourf(grids[mask,1], grids[mask,0], vdf_super[mask], levels=np.linspace(0,4.0,10), cmap='plasma')

        # ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, color='k', marker='.', s=3)
        plt.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan, c=(gvdf.vdfdata), s=50, cmap='plasma', norm=norm)
        # plt.scatter(-gvdf.vperp_nonan, gvdf.vpara_nonan, c=(gvdf.vdfdata), s=50, cmap='plasma', norm=norm)          # Only for testing
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(r'$v_{\perp}$ [km/s]', fontsize=19)
    ax.set_ylabel(r'$v_{\parallel}$ [km/s]', fontsize=19)
    ax.text(0.95, 0.95, r'$\mu = $' + f'{mu:.2e}', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'),
            transform=ax.transAxes, ha='right', va='top')
    ax.set_title(f'Super Resolution | {str(gvdf.l2_time[tidx])[:19]}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.set_xlim([-400,400])
    ax.set_aspect('equal')

    if SAVE:
        plt.savefig(f'./Figures/super_res_polcap/super_resolved_{tidx}_{gvdf.nptsx}_{gvdf.nptsy}.{ext}')
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
        
    # ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf.vshift[tidx], color='k', marker='.', s=3)
    ax.scatter(gvdf.vperp_nonan, gvdf.vpara_nonan - gvdf.vshift, color='k', marker='.', s=3)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=18) 
    ax.set_xlabel(r'$v_{\perp}$ [km/s]', fontsize=19)
    ax.set_ylabel(r'$v_{\parallel}$ [km/s]', fontsize=19)
    ax.set_title(f'Super Resolution | {str(gvdf.l2_time[tidx])[:19]}', fontsize=19)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim([-400,400])
    ax.set_aspect('equal')

    if(SAVE):
        plt.close()

def plot_Lcurve_knee_POLCAP(tidx, model_misfit, data_misfit, knee_idx, mu, ext='png', SAVE=False):
    fig = plt.figure()
    plt.plot(model_misfit, data_misfit, 'b')
    plt.plot(model_misfit, data_misfit, 'or')
    plt.plot(model_misfit[knee_idx], data_misfit[knee_idx], 'xk')
    plt.gca().text(0.95, 0.95, r'$\mu = $' + f'{mu:.2e}',
                   bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'),
                   transform=plt.gca().transAxes, ha='right', va='top')
    plt.grid(True)
    plt.xlabel('Model Misfit', fontsize=14, fontweight='bold')
    plt.ylabel('Data Misfit', fontsize=14, fontweight='bold')

    if(SAVE):
        plt.savefig(f'./Figures/kneeL_polcap/kneeL_{tidx}.{ext}')
        plt.close(fig)

def plot_CartSlep(xx, yy, CartSlep, tidx, ext='png', SAVE=False):
    # making the alpha arrays for even and odd functions
    # ea = even alpha, oa = odd alpha
    ea, oa = 1.0, 0.3
    alpha_arr = np.array([ea, ea, oa,
                          oa, ea, ea,
                          oa, ea, ea,
                          oa, ea, oa])

    # plotting the basis functions inside the domain
    fig, ax = plt.subplots(4, 3, figsize=(4,6.2), sharex=True, sharey=True)
    for i in range(12):
        row, col = i // 3, i % 3
        ax[row,col].pcolormesh(xx, yy, np.reshape(CartSlep.H[:,i], (49,49), 'C'),
                            cmap='seismic', rasterized=True, alpha=alpha_arr[i])
        ax[row,col].plot(CartSlep.XY[:,0], CartSlep.XY[:,1], 'k', alpha=alpha_arr[i])
        ax[row,col].set_aspect('equal')
        ax[row,col].set_title(r'$\mathbf{\alpha_{%i}}$='%(i+1) + f'{CartSlep.V[i]:.3f}', fontsize=10)
        ax[row,col].tick_params(axis='both', labelsize=7)
        ax[row,col].tick_params(axis='both', labelsize=7)

    plt.subplots_adjust(top=0.97, bottom=0.06, left=0.15, right=0.98, wspace=0.1, hspace=0.1)

    ax[3,1].set_xlabel(r'$v_{\perp}$ [km/s]', fontsize=12)
    fig.supylabel(r'$v_{\parallel}$ [km/s]', fontsize=12)

    if(SAVE):
        plt.savefig(f'Figures/cartesian_slepians/basis_tidx={tidx}.{ext}')
        plt.close()

def plot_super_resolution_CARTSLEP(gvdf_tstamp, CartSlep, xx, yy, f_data, f_supres, tidx, ext='png', SAVE=False):
    f_supres = np.reshape(f_supres, (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy)).T.flatten()

    # the SPAN data grids in FAC
    span_gridx = np.append(-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vperp_nonan)
    span_gridy = np.append(gvdf_tstamp.vpara_nonan, gvdf_tstamp.vpara_nonan)
    xmagmax = span_gridx.max() * 1.12

    Nspangrids = len(span_gridx)

    # making the colorbar norm function
    cmap = plt.cm.plasma
    lvls = np.linspace(int(np.log10(gvdf_tstamp.minval[tidx]) - 1),
                       int(np.log10(gvdf_tstamp.maxval[tidx]) + 1), 10)
    norm = colors.BoundaryNorm(lvls, ncolors=cmap.N)

    # plotting the points and the boundary
    fig, ax = plt.subplots(2, 1, figsize=(4.7,7.5), sharey=True)
    ax[0].plot(CartSlep.XY[:,0], CartSlep.XY[:,1], '--k')
    ax[0].scatter(span_gridx, span_gridy, c=np.log10(f_data), s=50,
                  cmap='plasma', norm=norm, edgecolor='k', linewidths=0.5)
    ax[0].set_aspect('equal')
    ax[0].set_xlim([-xmagmax, xmagmax])
    ax[0].text(0.02, 0.94, "(A)", transform=ax[0].transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.7))

    ax[1].plot(CartSlep.XY[:,0], CartSlep.XY[:,1], '--w')
    im = ax[1].tricontourf(xx.flatten(), yy.flatten(), np.log10(f_supres), levels=lvls, cmap='plasma')
    ax[1].scatter(span_gridx[Nspangrids//2:], span_gridy[Nspangrids//2:], c=np.log10(f_data[Nspangrids//2:]), s=50,
                  cmap='plasma', norm=norm)#, edgecolor='k', linewidths=0.5)
    ax[1].set_aspect('equal')
    ax[1].set_xlim([-xmagmax, xmagmax])
    ax[1].text(0.02, 0.94, "(B)", transform=ax[1].transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.7))

    ax[1].set_xlabel(r'$v_{\perp}$ [km/s]', fontsize=19)
    fig.supylabel(r'$v_{\parallel}$ [km/s]', fontsize=19)

    cax = fig.add_axes([ax[0].get_position().x0 + 0.06, ax[0].get_position().y1+0.05,
                        ax[0].get_position().x1 - ax[0].get_position().x0, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', location='top')
    cbar.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(integer=True)
    cbar.locator = tick_locator
    cbar.update_ticks()

    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.14, right=1.0, wspace=0.1, hspace=0.15)

    if(SAVE):
        plt.savefig(f'Figures/super_res_cartesian/tidx={tidx}.{ext}')
        plt.close()

def polcap_plotter(gvdf_tstamp, vdf_inv, vdf_super, tidx,
                   model_misfit=None, data_misfit=None, GRID=True, SAVE_FIGS=False):
    plot_span_vs_rec_contour_POLCAP(gvdf_tstamp, gvdf_tstamp.vdfdata, vdf_inv, tidx,
                                    GRID=True, SAVE=SAVE_FIGS)
    plot_super_resolution_POLCAP(gvdf_tstamp, vdf_super, gvdf_tstamp.mu_arr[gvdf_tstamp.knee_idx],
                                 tidx, VDFUNITS=True, VSHIFT=gvdf_tstamp.vel, SAVE=SAVE_FIGS)
    plot_Lcurve_knee_POLCAP(tidx, model_misfit, data_misfit, gvdf_tstamp.knee_idx,
                            gvdf_tstamp.mu_arr[gvdf_tstamp.knee_idx], SAVE=SAVE_FIGS)

def cartesian_plotter(gvdf_tstamp, vdf_inv, vdf_super, tidx,
                      model_misfit=None, data_misfit=None, GRID=True, SAVE_FIGS=False):
    # reshaping grids for plotting
    xx = np.reshape(gvdf_tstamp.grid_points[:,0], (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy), 'F')
    yy = np.reshape(gvdf_tstamp.grid_points[:,1], (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy), 'F')

    # converting the VDFs to SPAN-i consistent units
    f_supres = np.power(10, vdf_super) * gvdf_tstamp.minval[tidx]
    vdf_data = np.append(gvdf_tstamp.vdfdata, gvdf_tstamp.vdfdata)
    f_data = np.power(10, vdf_data) * gvdf_tstamp.minval[tidx]

    plot_super_resolution_CARTSLEP(gvdf_tstamp, gvdf_tstamp.CartSlep, xx, yy, f_data, f_supres, tidx, SAVE=SAVE_FIGS)

def hybrid_plotter(gvdf_tstamp, vdf_inv, vdf_super, tidx,
                   model_misfit=None, data_misfit=None, GRID=True, SAVE_FIGS=False, ext='png'):
    vdf_super_polcap, vdf_super_cartesian = vdf_super

    # converting the VDFs to SPAN-i consistent units
    f_supres_polcap = np.power(10, vdf_super_polcap) * gvdf_tstamp.minval[tidx]
    f_supres_cartesian = np.power(10, vdf_super_cartesian) * gvdf_tstamp.minval[tidx]

    # reshaping the VDFs correctly
    f_supres_A = np.reshape(f_supres_polcap, (gvdf_tstamp.nptsx,gvdf_tstamp.nptsy), 'F').T.flatten()
    f_supres_B = np.reshape(f_supres_cartesian, (gvdf_tstamp.nptsx,gvdf_tstamp.nptsy), 'F').T.flatten()

    # the SPAN data
    f_data = np.power(10, gvdf_tstamp.vdfdata) * gvdf_tstamp.minval[tidx]

    # the SPAN data grids in FAC
    span_gridx = np.append(-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vperp_nonan)
    span_gridy = np.append(gvdf_tstamp.vpara_nonan, gvdf_tstamp.vpara_nonan)
    xmagmax = span_gridx.max() * 1.12

    Nspangrids = len(span_gridx)
    
    # making the colorbar norm function
    cmap = plt.cm.plasma
    # lvls = np.linspace(int(np.log10(gvdf_tstamp.minval[tidx]) - 1),
    #                    int(np.log10(gvdf_tstamp.maxval[tidx]) + 1), 10)
    lvls = np.linspace(-22, -15, 10)
    norm = colors.BoundaryNorm(lvls, ncolors=cmap.N)

    # reshaping grids for plotting
    xx = np.reshape(gvdf_tstamp.grid_points[:,0], (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy), 'F')
    yy = np.reshape(gvdf_tstamp.grid_points[:,1], (gvdf_tstamp.nptsx, gvdf_tstamp.nptsy), 'F')

    # plotting the points and the boundary
    fig, ax = plt.subplots(2, 1, figsize=(4.7,9.5), sharey=True)
    ax[0].plot(gvdf_tstamp.CartSlep.XY[:,0], gvdf_tstamp.CartSlep.XY[:,1], '--w')
    im = ax[0].tricontourf(xx.flatten(), yy.flatten(), np.log10(f_supres_A), levels=lvls, cmap='plasma')
    ax[0].scatter(span_gridx[Nspangrids//2:], span_gridy[Nspangrids//2:], c=np.log10(f_data), s=50,
                  cmap='plasma', norm=norm, edgecolor='k', linewidths=0.5)
    ax[0].set_aspect('equal')
    ax[0].set_xlim([-xmagmax, xmagmax])
    ax[0].text(0.02, 0.94, "(A)", transform=ax[0].transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.7))

    ax[1].plot(gvdf_tstamp.CartSlep.XY[:,0], gvdf_tstamp.CartSlep.XY[:,1], '--w')
    im = ax[1].tricontourf(xx.flatten(), yy.flatten(), np.log10(f_supres_B), levels=lvls, cmap='plasma')
    ax[1].scatter(span_gridx[Nspangrids//2:], span_gridy[Nspangrids//2:], c=np.log10(f_data), s=50,
                  cmap='plasma', norm=norm, edgecolor='k', linewidths=0.5)
    ax[1].set_aspect('equal')
    ax[1].set_xlim([-xmagmax, xmagmax])
    ax[1].text(0.02, 0.94, "(B)", transform=ax[1].transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.7))

    ax[1].set_xlabel(r'$v_{\perp}$ [km/s]', fontsize=19)
    fig.supylabel(r'$v_{\parallel}$ [km/s]', fontsize=19)

    ax[0].set_xlim([-230, 230])
    ax[1].set_xlim([-230, 230])
    ax[0].set_ylim([50, 650])
    ax[1].set_ylim([50, 650])

    cax = fig.add_axes([ax[0].get_position().x0 + 0.06, ax[0].get_position().y1+0.05,
                        ax[0].get_position().x1 - ax[0].get_position().x0, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', location='top')
    cbar.ax.tick_params(labelsize=14)
    tick_locator = ticker.MaxNLocator(integer=True)
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax[0].set_title(f'{str(gvdf_tstamp.l2_time[tidx])[:19]}', fontsize=19)

    plt.subplots_adjust(top=0.87, bottom=0.1, left=0.14, right=1.0, wspace=0.1, hspace=0.15)

    if(SAVE_FIGS):
        plt.savefig(f'Figures/super_res_hybrid/tidx={tidx}.{ext}')
        plt.close()