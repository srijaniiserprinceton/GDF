import numpy as np
import cdflib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import sys
plt.ion()

import gdf.src.functions as fn
from gdf.src.misc_funcs import read_pickle

plt.rcParams['font.size'] = 12

mass_p_kg = 1.6726219e-27

def generate_inst_grid(psp_vdf):
    # Define the SPAN-i grids
    e_inst = psp_vdf.energy.data[0,:,:,:]
    vel_inst = 13.85*np.sqrt(psp_vdf.energy.data[0,:,:,:])
    theta_inst = np.radians(psp_vdf.theta.data[0,:,:,:])
    phi_inst = np.radians(psp_vdf.phi.data[0,:,:,:])

    # Define the new grids 
    vx_inst = vel_inst * np.cos(theta_inst) * np.cos(phi_inst)
    vy_inst = vel_inst * np.cos(theta_inst) * np.sin(phi_inst)
    vz_inst = vel_inst * np.sin(theta_inst)

    return(e_inst, vel_inst, theta_inst, phi_inst, vx_inst, vy_inst, vz_inst)

def plot_slices_xy(vx_inst, vy_inst, vdf_inter):
    fig, ax = plt.subplots(3, 5, layout='constrained', figsize=(12, 12), sharex=True, sharey=True)
    for i in range(15):
        row = i // 5
        col = i % 5

        vidx, tidx, pidx = np.unravel_index(np.nanargmax(vdf_inter[i]), (32,8,8))
        ax[row, col].scatter(vx_inst[:,tidx,:], vy_inst[:,tidx,:], color='k', marker='.')
        ax1 = ax[row, col].contourf(vx_inst[:,tidx,:], vy_inst[:,tidx,:], np.log10(vdf_inter[i,:,tidx,:]), levels=np.linspace(-24, -17, 10), cmap='plasma')

def plot_slices_xz(vx_inst, vz_inst, vdf_inter):
    fig, ax = plt.subplots(4, 5, layout='constrained', figsize=(12, 12), sharex=True, sharey=True)
    for i in range(15):
        row = i // 5
        col = i % 5

        vidx, tidx, pidx = np.unravel_index(np.nanargmax(vdf_inter[i]), (32,8,8))
        ax[row, col].scatter(vx_inst[:,tidx,:], vz_inst[:,tidx,:], color='k', marker='.')
        ax1 = ax[row, col].contourf(vx_inst[:,tidx,:], vz_inst[:,tidx,:], np.log10(vdf_inter[i,:,tidx,:]), levels=np.linspace(-24, -17, 10), cmap='plasma')

def plot_slices(i):
    fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12,6))
    vidx, tidx, pidx = np.unravel_index(np.nanargmax(ds.vdf.data[i]), (32,8,8))
    ax[0].scatter(vx[:,tidx,:], vy[:,tidx,:], color='k', marker='.')
    ax1 = ax[0].contourf(vx[:,tidx,:], vy[:,tidx,:], np.log10(ds.vdf.data[i,:,tidx,:]), levels=np.linspace(-24, -17, 10), cmap='plasma')
    ax[1].scatter(vx[:,:,pidx], vz[:,:,pidx], color='k', marker='.')
    ax2 = ax[1].contourf(vx[:,:,pidx], vz[:,:,pidx], np.log10(ds.vdf.data[i,:,:,pidx]), levels=np.linspace(-24, -17, 10), cmap='plasma')
    ax[0].set_xlabel(r'$v_x$')
    ax[0].set_ylabel(r'$v_y$')
    ax[1].set_xlabel(r'$v_x$')
    ax[1].set_ylabel(r'$v_z$')
    plt.colorbar(ax2)
    plt.savefig(f'./Figures/slices/tidx_{i}.png')
    plt.close()

def plot_slices_ranges_test_2(indecies, thetas):
    fig, ax = plt.subplots(3,3, layout='constrained', figsize=(8,8), sharex=True, sharey=True)
    vidx, tidx, pidx = np.unravel_index(np.nanargmax(ds.vdf.data[indecies[0]]), (32,8,8))
    [ax[i//3, i%3].scatter(-vx[:,tidx,:], -vy[:,tidx,:], color='k', marker='.', alpha=0.3) for i in range(9)]

    # Origin where arrows will start
    arrow_origin = (500, -250)

    for i, n in enumerate(indecies):
        l, m = i//3, i%3
        ax[l,m].contourf(-vx[:,tidx,:], -vy[:,tidx,:], np.log10(ds.vdf.data[indecies[i],:,tidx,:]), levels=np.linspace(-23, -17, 8), alpha=1.0, cmap='Blues_r')

        B_direction = 400*np.array([np.cos(np.radians(thetas[n])), -np.sin(np.radians(thetas[n]))])
        # Plot the arrow from origin in the direction of B
        ax[l,m].quiver(
            arrow_origin[0], arrow_origin[1],     # X, Y start point
            B_direction[0], B_direction[1],       # U, V direction
            angles='xy', scale_units='xy', scale=1,
            color='red', width=0.01
        )

        # Optional: Mark origin
        ax[l,m].plot(arrow_origin[0], arrow_origin[1], 'rx')

        ax[l,m].text(
            0.95, 0.06, f'$\\theta$: {thetas[n]:.2f}$^\circ$',
            transform=ax[l,m].transAxes,  # use axes-relative coordinates
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
        )
    [ax[i//3, i%3].set_ylim([None, 300]) for i in range(9)]
    [ax[i//3, i%3].set_aspect('equal') for i in range(9)]
    fig.supxlabel(r'$V_x$-inst', fontsize=16)
    fig.supylabel(r'$V_y$-inst', fontsize=16)


    plt.savefig('./Figures/paper_plots/Test2_demo.png')

def plot_slices_ranges_test_1(indecies, thetas):
    fig, ax = plt.subplots(3,3, layout='constrained', figsize=(8,8), sharex=True, sharey=True)
    vidx, tidx, pidx = np.unravel_index(np.nanargmax(ds.vdf.data[indecies[0]]), (32,8,8))
    [ax[i//3, i%3].scatter(-vx[:,tidx,:], -vy[:,tidx,:], color='k', marker='.', alpha=0.3) for i in range(9)]

    # Origin where arrows will start
    # arrow_origin = (-500, 250)

    for i, n in enumerate(indecies):
        l, m = i//3, i%3
        ax[l,m].contourf(-vx[:,tidx,:], -vy[:,tidx,:], np.log10(ds.vdf.data[indecies[i],:,tidx,:]), levels=np.linspace(-23, -17, 8), alpha=1.0, cmap='Blues_r')

        arrow_origin = [-ds.u_span.data[n,0], -ds.u_span.data[n,1]]
        B_direction = 400*np.array([np.cos(np.radians(thetas[n])), -np.sin(np.radians(thetas[n]))])
        # Plot the arrow from origin in the direction of B
        ax[l,m].quiver(
            arrow_origin[0], arrow_origin[1],     # X, Y start point
            B_direction[0], B_direction[1],       # U, V direction
            angles='xy', scale_units='xy', scale=1,
            color='red', width=0.01
        )

        # Optional: Mark origin
        ax[l,m].plot(arrow_origin[0], arrow_origin[1], 'rx')

        ax[l,m].text(
            0.95, 0.06, f'$\\theta$: {thetas[n]:.2f}$^\circ$',
            transform=ax[l,m].transAxes,  # use axes-relative coordinates
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
        )
    [ax[i//3, i%3].set_ylim([None, 300]) for i in range(9)]
    [ax[i//3, i%3].set_aspect('equal') for i in range(9)]
    fig.supxlabel(r'$V_x$-inst', fontsize=16)
    fig.supylabel(r'$V_y$-inst', fontsize=16)

    plt.savefig('./Figures/paper_plots/Test1_demo.png')

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

# Load in the hybrid and synthetic test cases
CASE = '2'

if CASE == '1':
    ds = cdflib.cdf_to_xarray('bimax_vdf_500_bhat_clockwise_field_rotation_pi_8_to_neg_pi_2_hires.cdf')
if CASE == '2':
    ds = cdflib.cdf_to_xarray('bimax_vdf_500_250_clockwise_field_rotation_corrected_hires.cdf')
# All of these figures are on the exact same grids!
energy, vel, theta, phi, vx, vy, vz = generate_inst_grid(ds)

ns = []
us = []
ts = []
for i in range(len(ds.time.data)):
    # n, ux, uu, uz = (fn.span_intergration(vel, theta, phi, vx, vy, vz, ds.vdf.data[i]))
    n, u, t_tens, _ = (fn.compute_vdf_moments(energy[:,0,0], (np.pi/2. - theta[0,:,0]), (phi[0,0,:]), ds.vdf.data[i], mass_p_kg))
    ns.append(n)
    us.append(u)
    ts.append(t_tens)



if CASE == '1':
    res_polcap = read_pickle('/home/michael/Research/GDF/synthetic_testing/Outputs/TEST1_vdf_rec_dataA_polcap_3_200_bimax_test_case')
    res_cart = read_pickle('/home/michael/Research/GDF/synthetic_testing/Outputs/TEST1_vdf_rec_dataA_cartesian_3_200_bimax_test_case')
    res_hybrid = read_pickle('/home/michael/Research/GDF/synthetic_testing/Outputs/TEST1_vdf_rec_dataA_hybrid_3_200_bimax_test_case')
if CASE == '2':
    res_polcap = read_pickle('/home/michael/Research/GDF/synthetic_testing/Outputs/TEST2_vdf_rec_dataA_polcap_3_200_bimax_test_case')
    res_cart = read_pickle('/home/michael/Research/GDF/synthetic_testing/Outputs/TEST2_vdf_rec_dataA_cartesian_3_200_bimax_test_case')
    res_hybrid = read_pickle('/home/michael/Research/GDF/synthetic_testing/Outputs/TEST2_vdf_rec_dataA_hybrid_3_200_bimax_test_case')

den_polcap = np.array([res_polcap[i]['den'] for i in res_polcap.keys()])
den_cart = np.array([res_cart[i]['den'] for i in res_cart.keys()])
den_hybrid = np.array([res_hybrid[i]['den'] for i in res_hybrid.keys()])

vel_polcap = np.array([res_polcap[i]['u_final'] for i in res_polcap.keys()])
vel_cart = np.array([res_cart[i]['u_final'] for i in res_cart.keys()])
vel_hybrid = np.array([res_hybrid[i]['u_final'] for i in res_hybrid.keys()])

t_comps_polcap = np.array([res_polcap[i]['component_temp'] for i in res_polcap.keys()])
t_comps_cart   = np.array([res_cart[i]['component_temp'] for i in res_cart.keys()])
t_comps_hybrid = np.array([res_hybrid[i]['component_temp'] for i in res_hybrid.keys()])

R = np.array([bvector_rotation_matrix(ds.b_span.data[i]) for i in range(100)])
ts_fa = np.array([R[i] @ ts[i] @ R[i].T for i in range(100)])
tpara = ts_fa[:,0,0]
tperp = (ts_fa[:,1,1] + ts_fa[:,2,2])/2

# Create figure with constrained layout
fig = plt.figure(constrained_layout=True, figsize=(6, 10))

# Define height ratios: top 3 panels = 1 unit each, bottom 3 = 2 units each
height_ratios = [1, 1, 1, 2, 2, 2]

# Create 6-row, 1-column grid
gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=height_ratios)

LW = 2
ALP = 0.6
if CASE == '1':
    thetas = -np.degrees(np.linspace(np.pi/8, -np.pi/2, 100))

    # plot_slices_ranges_test_1([0, 12, 25, 37, 50, 62, 75, 87, 99], thetas)
    # thetas = np.hstack([thetas, -np.pi/2])
if CASE == '2':
    thetas = np.degrees(np.linspace(-np.pi/2, np.pi/2, 100))
    # plot_slices_ranges_test_2([0, 12, 25, 37, 50, 62, 75, 87, 99], thetas)

colors = [np.array([230,159,0])/255, np.array([86,180,233])/255, np.array([0,158,115])/255, np.array([204, 121, 167])/255]
# colors = ['tab:blue', 'tab:orange', 'tab:red']

ax = []
for i in range(6):
    if i == 0:
        axs = fig.add_subplot(gs[i])
    else:
        axs = fig.add_subplot(gs[i], sharex=ax[0])
    ax.append(axs)


ax[0].plot(thetas, (vel_polcap[:,0]+1)/(ds.u_span.data[:,0]+1), marker='o', color=colors[0], lw=LW, alpha=ALP, label='Polar Cap')
ax[0].plot(thetas, (vel_cart[:,0]+1)/(ds.u_span.data[:,0]+1), marker='s', color=colors[1], lw=LW, alpha=ALP, label='Cartesian')
ax[0].plot(thetas, (vel_hybrid[:,0]+1)/(ds.u_span.data[:,0]+1), marker='v', color=colors[2], lw=LW, alpha=ALP, label='Hybrid')
ax[0].plot(thetas, (np.array(us)[:,0]+10)/(ds.u_span.data[:,0]+10), marker='.', color='k', lw=LW, alpha=ALP, label='SPAN-Ai Moment')
ax[0].legend(ncols=4, fontsize=8, frameon=False)

ax[1].plot(thetas, (vel_polcap[:,1]+1)/(ds.u_span.data[:,1]+1), marker='o', color=colors[0], lw=LW, alpha=ALP)
ax[1].plot(thetas, (vel_cart[:,1]+1)/(ds.u_span.data[:,1]+1), marker='s', color=colors[1], lw=LW, alpha=ALP)
ax[1].plot(thetas, (vel_hybrid[:,1]+1)/(ds.u_span.data[:,1]+1), marker='v', color=colors[2], lw=LW, alpha=ALP)
ax[1].plot(thetas, (np.array(us)[:,1]+10)/(ds.u_span.data[:,1]+10), marker='.', color='k', lw=LW, alpha=ALP)

if CASE == '1':
    ax[2].plot(thetas, (vel_polcap[:,2])/(ds.u_span.data[:,2]), marker='o', color=colors[0], lw=LW, alpha=ALP)
    ax[2].plot(thetas, (vel_cart[:,2])/(ds.u_span.data[:,2]), marker='s', color=colors[1], lw=LW, alpha=ALP)
    ax[2].plot(thetas, (vel_hybrid[:,2])/(ds.u_span.data[:,2]), marker='v', color=colors[2], lw=LW, alpha=ALP)
    ax[2].plot(thetas, (np.array(us)[:,2]+10)/(ds.u_span.data[:,2]+10), marker='.', color='k', lw=LW, alpha=ALP)
if CASE == '2':
    ax[2].plot(thetas, (vel_polcap[:,2]+100)/(ds.u_span.data[:,2]+100), marker='o', color=colors[0], lw=LW, alpha=ALP)
    ax[2].plot(thetas, (vel_cart[:,2]+100)/(ds.u_span.data[:,2]+100), marker='s', color=colors[1], lw=LW, alpha=ALP)
    ax[2].plot(thetas, (vel_hybrid[:,2]+100)/(ds.u_span.data[:,2]+100), marker='v', color=colors[2], lw=LW, alpha=ALP)
    ax[2].plot(thetas, (np.array(us)[:,2]+100)/(ds.u_span.data[:,2]+100), marker='.', color='k', lw=LW, alpha=ALP)



[ax[i].axhline(1, lw=4, alpha=1.0, color=colors[3], linestyle='dashed') for i in range(3)]

ax[3].plot(thetas, den_polcap, marker='o', color=colors[0], lw=LW, alpha=ALP, label='Polar Cap')
ax[3].plot(thetas, den_cart, marker='s', color=colors[1], lw=LW, alpha=ALP, label='Cartesian')
ax[3].plot(thetas, den_hybrid, marker='v', color=colors[2], lw=LW, alpha=ALP, label='Hybrid')
ax[3].plot(thetas, np.asarray(ns), marker='.', color='k', label='SPAN-Ai Moment', alpha=ALP)
ax[3].axhline(1702, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')

ax[4].plot(thetas, (t_comps_polcap[:,0]/573551.14), marker='o', color=colors[0], lw=LW, alpha=ALP)
ax[4].plot(thetas, (t_comps_cart[:,0]/573551.14), marker='s', color=colors[1], lw=LW, alpha=ALP)
ax[4].plot(thetas, (t_comps_hybrid[:,0]/573551.14), marker='v', color=colors[2], lw=LW, alpha=ALP)
ax[4].plot(thetas, tpara/573551.14, marker='.', color='k', label='SPAN-Ai Moment', alpha=ALP)
# ax[4].axhline(573551.14, lw=2, alpha=1.0, color='grey', linestyle='dashed')
ax[4].axhline(1, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')

ax[5].plot(thetas, t_comps_polcap[:,1]/943147.10, marker='o', color=colors[0], lw=LW, alpha=ALP)
ax[5].plot(thetas, t_comps_cart[:,1]/943147.10, marker='s', color=colors[1], lw=LW, alpha=ALP)
ax[5].plot(thetas, t_comps_hybrid[:,1]/943147.10, marker='v', color=colors[2], lw=LW, alpha=ALP)
# ax[5].axhline(943147.10, lw=2, alpha=1.0, color='grey', linestyle='dashed')
ax[5].plot(thetas, tperp/943147.10, marker='.', color='k', label='SPAN-Ai Moment', alpha=ALP)
ax[5].axhline(1, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')

ax[3].set_ylabel('Density\n'+r'[$cm^{-3}$]')
idx = ['x', 'y', 'z']
[ax[i].set_ylabel(f'$V_{idx[i]}/U_{idx[i]}$') for i in range(3)]
ax[4].set_ylabel(r'$\mathrm{T}_{\parallel}\left/\mathrm{T}_{\parallel}^{0}\right.$')
ax[5].set_ylabel(r'$\mathrm{T}_{\perp}\left/\mathrm{T}_{\perp}^{0}\right.$')

[ax[i].set_xlim([thetas[0], thetas[-1]]) for i in range(6)]
[ax[i].label_outer() for i in range(5)]

if CASE == '1':
    [ax[i].set_ylim([0.9, 1.5]) for i in range(3)]
    ax[3].set_ylim([-950, 2550])
    ax[4].set_ylim([0.4, 3.0])
    ax[5].set_ylim([0.4, 3.0])

    [ax[i].axvspan((thetas[0]), (thetas[46]), color='0.85') for i in range(6)]
    ax[5].set_xlabel('Angle (deg)')


    # --- Add SPC FOV arrow above the top subplot ---
    # Get axis limits to place arrow just above the top panel
    ax0 = ax[0]
    x_start = thetas[0]
    x_end = thetas[47]
    y = ax0.get_ylim()[1]  # top y-limit of the first panel

    # Slight padding for visual clarity
    arrow_y = y + 0.05*y
    text_y = y + 0.05*y

    # Draw double-headed arrow
    ax0.annotate(
        '', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
        arrowprops=dict(arrowstyle='<|-|>', lw=1.5, color='black'),
        annotation_clip=False
    )

    # Add centered label above the arrow
    # Add label in the center of the arrow with white background
    ax0.text(
        (x_start + x_end) / 2, text_y, 'SPC FOV',
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', edgecolor='none', pad=2)
    )

    # # --- Inset configuration ---
    # 1. Create an inset axis within ax
    ax_inset3 = inset_axes(ax[3], width="69.5%", height="35%", loc='lower right',
                        bbox_to_anchor=(0.0, 0.15, 1.0, 1.0), bbox_transform=ax[3].transAxes,
                        borderpad=0.0)

    # 2. Plot the same data in the inset
    start_idx = 30
    end_idx = -1 
    ax_inset3.axvspan(thetas[start_idx], thetas[46], color='0.85')
    ax_inset3.axhline(1702, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')
    ax_inset3.plot(thetas[start_idx:end_idx], den_polcap[start_idx:end_idx], marker='o', color=colors[0], lw=LW, alpha=ALP)
    ax_inset3.plot(thetas[start_idx:end_idx], den_cart[start_idx:end_idx], marker='s', color=colors[1], lw=LW, alpha=ALP)
    ax_inset3.plot(thetas[start_idx:end_idx], den_hybrid[start_idx:end_idx], marker='v', color=colors[2], lw=LW, alpha=ALP)
    ax_inset3.plot(thetas[start_idx:end_idx], np.asarray(ns)[start_idx:end_idx], marker='.', color='k', label='SPAN-Ai Moment', alpha=ALP)
    


    # 3. Define zoomed-in range
    x1, x2 = thetas[start_idx], thetas[end_idx]  # angle range
    y1, y2 = 1620, 1860  # y range

    xticks = ax_inset3.get_xticks()
    # Keep every other label (e.g., even indices)
    xtick_labels = [f"{tick:.0f}" if i % 2 == 0 else "" for i, tick in enumerate(xticks[0:-1])]

    # Apply the new tick labels
    ax_inset3.set_xticklabels(xtick_labels[1:-1])

    ax_inset3.set_xlim(x1, x2)
    ax_inset3.set_ylim(y1, y2)
    ax_inset3.tick_params(labelsize=12)
    ax_inset3.axhline(1702, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')


    # # --- Inset configuration ---
    # 1. Create an inset axis within ax
    ax_inset4 = inset_axes(ax[4], width="69.5%", height="50%", loc='upper right',
                        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=ax[4].transAxes,
                        borderpad=0.0)

    # 2. Plot the same data in the inset
    start_idx = 30
    end_idx = -1 
    ax_inset4.axvspan(thetas[start_idx], thetas[46], color='0.85')
    ax_inset4.axhline(1702, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')
    ax_inset4.plot(thetas[start_idx:end_idx], (t_comps_polcap[start_idx:end_idx,0]/573551.14), marker='o', color=colors[0], lw=LW, alpha=ALP)
    ax_inset4.plot(thetas[start_idx:end_idx], (t_comps_cart[start_idx:end_idx,0]/573551.14), marker='s', color=colors[1], lw=LW, alpha=ALP)
    ax_inset4.plot(thetas[start_idx:end_idx], (t_comps_hybrid[start_idx:end_idx,0]/573551.14), marker='v', color=colors[2], lw=LW, alpha=ALP)
    ax_inset4.plot(thetas[start_idx:end_idx], tpara[start_idx:end_idx]/573551.14, marker='.', color='k', label='SPAN-Ai Moment', alpha=ALP)

    # 3. Define zoomed-in range
    y1_1, y2_1 = 0.8, 1.05  # y range

    # Apply the new tick labels
    ax_inset4.set_xticklabels(xtick_labels[1:-1])

    ax_inset4.set_xlim(x1, x2)
    ax_inset4.set_ylim(y1_1, y2_1)
    ax_inset4.tick_params(labelsize=12)
    ax_inset4.axhline(1, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')


    # # --- Inset configuration ---
    # 1. Create an inset axis within ax
    ax_inset5 = inset_axes(ax[5], width="69.5%", height="50%", loc='upper right',
                        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=ax[5].transAxes,
                        borderpad=0.0)

    # 2. Plot the same data in the inset
    start_idx = 30
    end_idx = -1 
    ax_inset5.axvspan(thetas[start_idx], thetas[46], color='0.85')
    ax_inset5.axhline(1702, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')
    ax_inset5.plot(thetas[start_idx:end_idx], (t_comps_polcap[start_idx:end_idx,1]/943147.10), marker='o', color=colors[0], lw=LW, alpha=ALP)
    ax_inset5.plot(thetas[start_idx:end_idx], (t_comps_cart[start_idx:end_idx,1]/943147.10), marker='s', color=colors[1], lw=LW, alpha=ALP)
    ax_inset5.plot(thetas[start_idx:end_idx], (t_comps_hybrid[start_idx:end_idx,1]/943147.10), marker='v', color=colors[2], lw=LW, alpha=ALP)
    ax_inset5.plot(thetas[start_idx:end_idx], tperp[start_idx:end_idx]/943147.10, marker='.', color='k', label='SPAN-Ai Moment', alpha=ALP)

    # 3. Define zoomed-in range
    y1_1, y2_1 = 0.8, 1.05  # y range

    # Apply the new tick labels
    ax_inset5.set_xticklabels(xtick_labels[1:-1])

    ax_inset5.set_xlim(x1, x2)
    ax_inset5.set_ylim(y1_1, y2_1)
    ax_inset5.tick_params(labelsize=12)
    ax_inset5.axhline(1, lw=4, alpha=1.0, color=colors[3], linestyle='dashed')

    plt.savefig('./Figures/paper_plots/case_1_figure.png')


if CASE == '2':
    [ax[i].set_ylim([0.95, 1.1]) for i in range(3)]
    ax[5].set_xlabel('Angle (deg)')

    [ax[i].axvspan(0, 45, color='0.85') for i in range(6)]

    # --- Add SPC FOV arrow above the top subplot ---
    # Get axis limits to place arrow just above the top panel
    ax0 = ax[0]
    x_start = 0
    x_end = 45
    y = ax0.get_ylim()[1]  # top y-limit of the first panel

    # Slight padding for visual clarity
    arrow_y = y + 0.03*y
    text_y = y + 0.03*y

    # Draw double-headed arrow
    ax0.annotate(
        '', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
        arrowprops=dict(arrowstyle='<|-|>', lw=1.5, color='black'),
        annotation_clip=False
    )

    # Add centered label above the arrow
    # Add label in the center of the arrow with white background
    ax0.text(
        (x_start + x_end) / 2, text_y, 'Typical B-field',
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', edgecolor='none', pad=2)
    )

    plt.savefig('./Figures/paper_plots/case_2_figure.png')