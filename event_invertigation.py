from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import numpy as np
import cdflib 
import xarray as xr
import functions as fn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.ion()

def plot_moments_data(mom, CLIP=None):
    # get the velocity and magnetic field data.
    vel_data = mom.VEL_INST.data
    mag_data = mom.MAGF_INST.data 

    vel = np.linalg.norm(vel_data, axis=1)
    mag = np.linalg.norm(mag_data, axis=1)

    theta_vb = np.degrees(np.arccos(np.einsum('ij, ij->i', vel_data, mag_data)/(vel * mag)))

    fig, ax = plt.subplots(4, figsize=(12, 8), sharex=True)

    [ax[0].plot(mom.Epoch.data, mom.MAGF_INST[:,i].data) for i in range(3)]
    [ax[1].plot(mom.Epoch.data, mom.VEL_INST[:,i].data) for i in range(3)]
    ax[2].plot(mom.Epoch.data, mom.DENS.data, color='k')
    ax[3].plot(mom.Epoch.data, theta_vb, color='k')

    ax[3].set_ylim([-180, 180])
    ax[3].axhline(0, linestyle='dashed', color='grey')
    ax[3].axhline(90, linestyle='dotted', color='grey')
    ax[3].axhline(-90, linestyle='dotted', color='grey')

    if CLIP:
        [ax[i].set_xlim(trange[0], trange[1]) for i in range(4)]

    plt.show()
    
def plot_moments_data_field_aligned_coordinates(mom, CLIP=None):
    # get the velocity and magnetic field data.
    vel_data = mom.VEL_INST.data
    mag_data = mom.MAGF_INST.data 

    v_fac = np.array([fn.rotate_vector_field_aligned(*vel_data[i], *fn.field_aligned_coordinates(mag_data[i])) for i in range(len(vel_data))])
    b_fac = np.array([fn.rotate_vector_field_aligned(*mag_data[i], *fn.field_aligned_coordinates(mag_data[i])) for i in range(len(vel_data))])

    vel = np.linalg.norm(vel_data, axis=1)
    mag = np.linalg.norm(mag_data, axis=1)

    theta_vb = np.degrees(np.arccos(np.einsum('ij, ij->i', vel_data, mag_data)/(vel * mag)))

    fig, ax = plt.subplots(4, figsize=(12, 8), sharex=True)

    [ax[0].plot(mom.Epoch.data, v_fac[:,i]) for i in range(3)]
    [ax[1].plot(mom.Epoch.data, b_fac[:,i]) for i in range(3)]
    ax[2].plot(mom.Epoch.data, mom.DENS.data, color='k')
    ax[3].plot(mom.Epoch.data, theta_vb, color='k')

    ax[3].set_ylim([-180, 180])
    ax[3].axhline(0, linestyle='dashed', color='grey')
    ax[3].axhline(90, linestyle='dotted', color='grey')
    ax[3].axhline(-90, linestyle='dotted', color='grey')

    if CLIP:
        [ax[i].set_xlim(trange[0], trange[1]) for i in range(4)]

    plt.show()

def plot_span_bimax_data(mom, bimax):
    # get the velocity and magnetic field data.
    mom_time = mom.Epoch.data
    den_data = mom.DENS.data
    vel_data = mom.VEL_INST.data
    mag_data = mom.MAGF_INST.data

    fit_time = bimax.Epoch.data
    fit_den_data = bimax.n_tot.data
    fit_vel_data = bimax.vcm.data

    fig, ax = plt.subplots(6, figsize=(8,16), sharex=True)
    ax[0].plot(mom_time, den_data, color='k')
    ax[0].plot(fit_time, fit_den_data, alpha=0.5)
    ax[0].set_ylim([np.nanmin(den_data), np.nanmax(den_data)])
    [ax[i+1].plot(mom_time, vel_data[:,i], color='k') for i in range(3)]
    [ax[i+1].plot(fit_time, fit_vel_data[:,i], alpha=0.5) for i in range(3)]
    [ax[i+1].set_ylim([np.nanmin(vel_data[:,i]), np.nanmax(vel_data[:,i])]) for i in range(3)]

    [ax[i].set_xlim([np.datetime64('2024-12-24T10:00:00'), np.datetime64('2024-12-24T10:05:00')]) for i in range(6)]

    plt.show()


def plot_eflux_panels(mom):
    def get_bin_edges(vals):
        edges = np.zeros(len(vals) + 1)
        edges[1:-1] = 0.5 * (vals[1:] + vals[:-1])
        edges[0] = vals[0] - (vals[1] - vals[0]) / 2
        edges[-1] = vals[-1] + (vals[-1] - vals[-2]) / 2
        return edges

    # --- Data unpacking ---
    time = mom.Epoch.data.astype('datetime64[s]')
    time_sec = time.astype('int64')
    time_mid = (time_sec[:-1] + time_sec[1:]) // 2
    first_edge = time_sec[0] - (time_mid[0] - time_sec[0])
    last_edge = time_sec[-1] + (time_sec[-1] - time_mid[-1])
    time_edges_sec = np.concatenate([[first_edge], time_mid, [last_edge]])
    time_edges_datetime = time_edges_sec.astype('datetime64[s]').astype(object)
    time_edges_num = mdates.date2num(time_edges_datetime)

    energy_vals = mom.ENERGY_VALS.data[0]
    theta_vals = mom.THETA_VALS.data[0]
    phi_vals = mom.PHI_VALS.data[0]
    energy_edges = get_bin_edges(energy_vals)
    theta_edges = get_bin_edges(theta_vals)
    phi_edges = get_bin_edges(phi_vals)

    eflux_energy = np.stack(mom.EFLUX_VS_ENERGY.data)
    eflux_theta = np.stack(mom.EFLUX_VS_THETA.data)
    eflux_phi = np.stack(mom.EFLUX_VS_PHI.data)

    # --- Set up figure and axes ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'width_ratios': [1]}, sharex=True    )

    # ENERGY
    div0 = make_axes_locatable(axs[0])
    cax0 = div0.append_axes("right", size="3%", pad=0.05)
    pcm0 = axs[0].pcolormesh(time_edges_num, energy_edges, eflux_energy.T/np.nanmax(eflux_energy.T, axis=0)[None,:], cmap='inferno', shading='auto')
    axs[0].set_yscale('log')
    axs[0].set_ylabel("Energy [eV]", fontsize=22)
    plt.colorbar(pcm0, cax=cax0)

    # THETA
    div1 = make_axes_locatable(axs[1])
    cax1 = div1.append_axes("right", size="3%", pad=0.05)
    pcm1 = axs[1].pcolormesh(time_edges_num, theta_edges, eflux_theta.T/np.nanmax(eflux_theta.T, axis=0)[None,:], cmap='inferno', shading='auto')
    axs[1].set_ylabel("Theta [deg]", fontsize=22)
    plt.colorbar(pcm1, cax=cax1)

    # PHI
    div2 = make_axes_locatable(axs[2])
    cax2 = div2.append_axes("right", size="3%", pad=0.05)
    pcm2 = axs[2].pcolormesh(time_edges_num, phi_edges, eflux_phi.T/np.nanmax(eflux_phi.T, axis=0)[None,:], cmap='inferno', shading='auto')
    axs[2].set_ylabel("Phi [deg]", fontsize=22)
    plt.colorbar(pcm2, cax=cax2)

    div3 = make_axes_locatable(axs[3])
    cax3 = div3.append_axes("right", size="3%", pad=0.05)
    label = [r'$B_{x}$', r'$B_{y}$', r'$B_{z}$']
    [axs[3].plot(time, mom.MAGF_INST.data[:,i], label=label[i]) for i in range(3)]
    axs[3].set_ylabel(r"$B_{INST}$", fontsize=22)
    axs[3].set_xlabel("Time [UTC]", fontsize=22)
    axs[3].legend(ncols=3, fontsize=20, frameon=False)
    cax3.set_xticks([])
    cax3.set_yticks([])

    # Format time axis
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)

    plt.show()


if __name__ == "__main__":
    trange = ['2024-12-25T00:00:00', '2024-12-25T23:59:59']
    credentials = fn.load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]

    # get the moments.
    mom_data = fn.init_psp_moms(trange, CREDENTIALS=creds, CLIP=True)
    
    # get the biMax fits
    cdfdata = cdflib.cdf_to_xarray('/home/michael/Research/GDF/biMax_Fits/spp_swp_spi_sf00_fits_2024-12-24_v00.cdf', to_datetime=True)



    