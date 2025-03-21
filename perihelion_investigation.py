import json
import numpy as np
import scipy as sp

import functions as fn
import coordinate_frame_functions as coor_fn
import inversion as inv

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider

import astropy.constants as c
import astropy.units as u

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def plot_vdf_slices(vdf_ds, FRAME='INST', U_VEC=None, B_VEC=None, SUM=False, PLOT_TYPE='pcolormesh', POINT=None):
    vdf = vdf_ds.vdf.data
    # Define the velocity
    velocity = 13.85 * np.sqrt(vdf_ds.energy.data)
    theta = vdf_ds.theta.data
    phi   = vdf_ds.phi.data

    # This will load in the VDF for a given timestamp
    vx = velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    vy = velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    vz = velocity * np.sin(np.radians(theta))

    if FRAME == 'PLASMA':
        if np.any(U_VEC):
            vx = vx - U_VEC[0]
            vy = vy - U_VEC[1]
            vz = vz - U_VEC[2]
        else:
            print("Now bulk speed. Defaulting to instrument frame")

    # Generate the VDF Slices
    fig, ax = plt.subplots(1,2, figsize=(12,6), layout='constrained')

    # Find the peak vdf index values
    vidx, tidx, pidx = np.unravel_index(np.nanargmax(vdf), vdf.shape)

    # Find max extent
    vdf[vdf == 0] = np.nan
    mask = np.isnan(vdf)
    max_vx = np.max(np.abs(vx[~mask]))
    max_vy = np.max(np.abs(vy[~mask]))
    max_vz = np.max(np.abs(vz[~mask]))

    if SUM:
        vxp1, vyp1 = vx[:,tidx,:], vy[:,tidx,:]
        vdfp1 = np.nansum(vdf, axis=1)

        vxp2, vzp2 = vx[:,:,pidx], vz[:,:,pidx]
        vdfp2 = np.nansum(vdf, axis=2)
    else:    
        vxp1, vyp1 = vx[:,tidx,:], vy[:,tidx,:]
        vdfp1 = vdf[:,tidx,:]

        vxp2, vzp2 = vx[:,:,pidx], vz[:,:,pidx]
        vdfp2 = vdf[:,:,pidx]
        
    if FRAME == 'Plasma':
        axis_label = ['Vx-Plasma (km/s)','Vy-Plasma (km/s)','Vz-Plasma (km/s)']
    else:
        axis_label = ['Vx-Inst (km/s)','Vy-Inst (km/s)','Vz-Inst (km/s)']


    if PLOT_TYPE == 'contourf':
        ax[0].contourf(vxp1, vyp1, vdfp1, norm=LogNorm(), cmap='plasma', levels=20)
        ax[1].contourf(vxp2, vzp2, vdfp2, norm=LogNorm(), cmap='plasma', levels=20)
    if PLOT_TYPE == 'pcolormesh':
        ax[0].pcolormesh(vxp1, vyp1, vdfp1, norm=LogNorm(), cmap='plasma')
        ax[1].pcolormesh(vxp2, vzp2, vdfp2, norm=LogNorm(), cmap='plasma')

    
    ax[0].scatter(vxp1, vyp1, marker='.', color='k', alpha=0.2)
    ax[0].set_title(r'Vx-Vy plane on $\theta$ = '+f'{theta[0,tidx,0]} Slice')
    ax[0].set_xlabel(axis_label[0])
    ax[0].set_ylabel(axis_label[1])
    ax[0].set_xlim([-1.2*np.max((max_vx, max_vy)), 0])
    ax[0].set_ylim([0, 1.2*np.max((max_vx, max_vy))])
    ax[0].scatter(U_VEC[0], U_VEC[1], marker='d', color='r', s=50, label=f'(Vx, Vy, Vz) = ({str(np.round(U_VEC[0], 1))}, {str(np.round(U_VEC[1], 1))}, {str(np.round(U_VEC[2], 1))})')

    if np.any(POINT):
        ax[0].scatter(POINT[0], POINT[1], marker='*', color='blue', s=50, label=f'(Vx, Vy, Vz) = ({str(np.round(POINT[0], 1))}, {str(np.round(POINT[1], 1))}, {str(np.round(POINT[2], 1))})')
        ax[1].scatter(POINT[0], POINT[2], marker='*', color='blue', s=50)

    ax[0].legend(fontsize=12)
    
    ax[1].scatter(vxp2, vzp2, marker='.', color='k', alpha=0.2)
    ax[1].set_xlabel(axis_label[0])
    ax[1].set_ylabel(axis_label[2])
    ax[1].set_title(r'Vx-Vz plane on $\phi$ = '+f'{phi[0,0,pidx]} Slice')
    ax[1].set_xlim([-1.2*np.max((max_vx, max_vz)), 0])
    ax[1].set_ylim([-1*np.max((max_vx, max_vz)), 1*np.max((max_vx, max_vz))])
    ax[1].scatter(U_VEC[0], U_VEC[2], marker='d', color='r', s=50)

    [ax[i].set_aspect('equal') for i in range(2)]
    
    if np.any(U_VEC) and np.any(B_VEC):
        ax[0].quiver(U_VEC[0], U_VEC[1], U_VEC[0] + B_VEC[0], U_VEC[1] + B_VEC[1], scale = 1, scale_units='xy')
        ax[1].quiver(U_VEC[0], U_VEC[2], U_VEC[0] + B_VEC[0], U_VEC[2] + B_VEC[2], scale = 1, scale_units='xy')
    
    
    
    plt.show()

def plot_vdf_slices_interactive(vdf_ds, FRAME='INST', U_VEC=None, B_VEC=None, SUM=False, PLOT_TYPE='pcolormesh'):
    vdf = vdf_ds.vdf.data
    velocity = 13.85 * np.sqrt(vdf_ds.energy.data)
    theta = vdf_ds.theta.data
    phi = vdf_ds.phi.data
    
    vx = velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    vy = velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    vz = velocity * np.sin(np.radians(theta))
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    def update(idx):
        vidx, tidx, pidx = np.unravel_index(np.nanargmax(vdf[idx]), vdf[idx].shape)
        mask = np.isnan(vdf[idx])
        max_vx = np.max(np.abs(vx[idx][~mask]))
        max_vy = np.max(np.abs(vy[idx][~mask]))
        max_vz = np.max(np.abs(vz[idx][~mask]))
        
        if FRAME == 'PLASMA' and U_VEC is not None:
            vx[idx] -= U_VEC[idx,0]
            vy[idx] -= U_VEC[idx,1]
            vz[idx] -= U_VEC[idx,2]
        
        ax[0].clear()
        ax[1].clear()

        if SUM:
            vxp1, vyp1 = vx[idx,:,tidx,:], vy[idx,:,tidx,:]
            vdfp1 = np.nansum(vdf[idx], axis=1)
            vxp2, vzp2 = vx[idx,:,:,pidx], vz[idx,:,:,pidx]
            vdfp2 = np.nansum(vdf[idx], axis=2)
        else:
            vxp1, vyp1 = vx[idx,:,tidx,:], vy[idx,:,tidx,:]
            vdfp1 = vdf[idx,:,tidx,:]
            vxp2, vzp2 = vx[idx,:,:,pidx], vz[idx,:,:,pidx]
            vdfp2 = vdf[idx,:,:,pidx]
        
        if FRAME == 'PLASMA':
            axis_label = ['Vx-Plasma (km/s)','Vy-Plasma (km/s)','Vz-Plasma (km/s)']
        else:
            axis_label = ['Vx-Inst (km/s)','Vy-Inst (km/s)','Vz-Inst (km/s)']
        
        if PLOT_TYPE == 'contourf':
            ax[0].contourf(vxp1, vyp1, vdfp1, norm=LogNorm(), cmap='plasma', levels=20)
            ax[1].contourf(vxp2, vzp2, vdfp2, norm=LogNorm(), cmap='plasma', levels=20)
        elif PLOT_TYPE == 'pcolormesh':
            ax[0].pcolormesh(vxp1, vyp1, vdfp1, norm=LogNorm(), cmap='plasma')
            ax[1].pcolormesh(vxp2, vzp2, vdfp2, norm=LogNorm(), cmap='plasma')
        
        ax[0].scatter(vxp1, vyp1, marker='.', color='k', alpha=0.2)
        ax[0].set_title(f'Vx-Vy plane on θ = {theta[idx,0,tidx,0]:.2f} Slice')
        ax[0].set_xlabel(axis_label[0])
        ax[0].set_ylabel(axis_label[1])
        ax[0].set_xlim([-1.*np.max((max_vx, max_vy)), 0])
        ax[0].set_ylim([0, 1.*np.max((max_vx, max_vy))])
        
        ax[1].scatter(vxp2, vzp2, marker='.', color='k', alpha=0.2)
        ax[1].set_title(f'Vx-Vz plane on φ = {phi[idx,0,0,pidx]:.2f} Slice')
        ax[1].set_xlabel(axis_label[0])
        ax[1].set_ylabel(axis_label[2])
        ax[1].set_xlim([-1.*np.max((max_vx, max_vz)), 0])
        ax[1].set_ylim([-1.*np.max((max_vx, max_vz)), 1.*np.max((max_vx, max_vz))])

        if np.any(U_VEC) and np.any(B_VEC):
            ax[0].quiver(U_VEC[idx,0], U_VEC[idx,1], U_VEC[idx,0] + B_VEC[idx,0], U_VEC[idx,1] + B_VEC[idx,1], scale = 1, scale_units='xy')
            ax[1].quiver(U_VEC[idx,0], U_VEC[idx,2], U_VEC[idx,0] + B_VEC[idx,0], U_VEC[idx,2] + B_VEC[idx,2], scale = 1, scale_units='xy')
    
        for i in range(2):
            ax[i].set_aspect('equal')
        
        plt.draw()
    
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time Index', 0, len(vdf_ds.time.data), valinit=0, valstep=1)
    slider.on_changed(lambda val: update(int(val)))
    
    update(tidx)
    fig.canvas.draw()


if __name__ == "__main__":
    # We are investigating the VDFs at perihelion
    trange = ['2024-12-24T16:00:00', '2024-12-24T18:00:00']
    # trange = ['2020-01-26T00:00:00', '2020-01-26T23:00:00']
    # trange = ['2020-01-29T00:00:00', '2020-01-29T23:00:00']
    # trange = ['2018-11-07T00:00:00', '2018-11-07T23:59:59']
    # Use the user credentials
    credentials = load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    # creds = None
    # psp_vdf = fn._get_psp_vdf(trange, CREDENTIALS=creds)
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=True)

    psp_moms = fn.init_psp_moms(trange, CREDENTIALS=creds, CLIP=True)

    # tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2024-12-24T20:43:46')))
    # tidx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T14:10:42')))
    tidx = 200
    # # Get the PSP Flags
    # peak_theta = np.nanargmax(psp_moms.EFLUX_VS_THETA.data, axis=1)
    # peak_phi   = np.nanargmax(psp_moms.EFLUX_VS_PHI.data, axis=1)
    density = psp_moms.DENS.data
    avg_den = np.convolve(density, np.ones(10)/10, 'same')      # 1-minute average

    va_vec = ((psp_moms.MAGF_INST.data * u.nT) / (np.sqrt(c.m_p * c.mu0 * avg_den[:,None] * u.cm**(-3)))).to(u.km/u.s).value

    u_vec = psp_moms.VEL_INST.data
    # plot_vdf_slices_interactive(psp_vdf, U_VEC=psp_moms.VEL_INST.data, B_VEC=va_vec, PLOT_TYPE='contourf', SUM=True)

    