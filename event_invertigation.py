from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import numpy as np
import cdflib 
import xarray as xr
import src.functions as fn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
plt.ion()

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

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
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'width_ratios': [1]}, sharex=True)

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
    axs[3].set_ylim([-1000,1000])

    # Format time axis
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)

    plt.show()


def plot_sbr_span_bimax(moms, sbr, biMax=None, lfr_den=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'width_ratios': [1]}, sharex=True)

    axs[0].plot(moms.Epoch.data, moms.DENS.data, color='k', alpha=0.6, lw=4, label='Moment')
    [axs[i+1].plot(moms.Epoch.data, moms.VEL_INST[:,i].data, color='k', alpha=0.6, lw=4) for i in range(3)]


    if biMax:
        mask = (biMax.n_tot.data > 2*moms.DENS.data)
        bm_den = biMax.n_tot.data
        bm_den[mask] = np.nan

        bm_vel = biMax.vcm.data
        bm_vel[mask] = np.nan
        axs[0].plot(biMax.Epoch.data, bm_den, color='tab:blue', alpha=0.8, lw=2, label='biMax')
        [axs[i+1].plot(moms.Epoch.data, bm_vel[:,i], color='tab:blue', alpha=0.8, lw=2) for i in range(3)]

    # Load in sbr density and velocity
    sbr_time = np.array([sbr[i]['time'] for i in sbr.keys()]) 
    sbr_den = np.array([sbr[i]['den'] for i in sbr.keys()])
    sbr_vel = np.array([sbr[i]['u_final'] for i in sbr.keys()])

    sbr_mask = sbr_den > 4000
    sbr_den[sbr_mask] = np.nan
    sbr_vel[sbr_mask] = np.nan

    nanmask = np.isnan(sbr_den)

    sbr_time = sbr_time[~nanmask]
    sbr_den = sbr_den[~nanmask]
    sbr_vel = sbr_vel[~nanmask]

    axs[0].plot(sbr_time, sbr_den, color='tab:red', alpha=1.0, lw=2, label='SBR')
    [axs[i+1].plot(sbr_time, sbr_vel[:,i], color='tab:red', alpha=1.0, lw=2) for i in range(3)]

    axs[0].set_ylabel(r'Density [$cm^{-3}$]', fontsize=22)
    axs[1].set_ylabel(r'$V_x$ [km/s]', fontsize=22)
    axs[2].set_ylabel(r'$V_y$ [km/s]', fontsize=22)
    axs[3].set_ylabel(r'$V_z$ [km/s]', fontsize=22)
    axs[3].set_xlabel('Time [UTC]', fontsize=22)

    axs[3].set_xlim([sbr_time[0], sbr_time[-1]])
    

    if lfr_den: 
        qtn_mask = lfr_den.electron_density.data < 0
        lfrden = lfr_den.electron_density.data
        lfrden[qtn_mask] = np.nan
        axs[0].plot(lfr_den.Epoch.data, lfr_den.electron_density.data, color='tab:orange', label = 'LFR QTN')

    axs[0].legend(ncols=4, fontsize=18, frameon=False)
    axs[0].set_ylim([0,1000])


    plt.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)

    plt.show()


def plot_sbr_span_bimax_density(moms, biMax, sbr, lfr_den=None):
    fig, axs = plt.subplots(figsize=(12, 8))

    axs.plot(moms.Epoch.data, moms.DENS.data, color='k', alpha=0.6, lw=4, label='Moment')

    mask = (biMax.n_tot.data > 2*moms.DENS.data)
    bm_den = biMax.n_tot.data
    bm_den[mask] = np.nan

    axs.plot(biMax.Epoch.data, bm_den, color='tab:blue', alpha=0.8, lw=2, label='biMax')

    # Load in sbr density and velocity
    sbr_time = np.array([sbr[i]['time'] for i in sbr.keys()]) 
    sbr_den = np.array([sbr[i]['den'] for i in sbr.keys()])

    sbr_mask = sbr_den > 4000
    sbr_den[sbr_mask] = np.nan

    nanmask = np.isnan(sbr_den)

    sbr_time = sbr_time[~nanmask]
    sbr_den = sbr_den[~nanmask]

    axs.plot(sbr_time, sbr_den, color='tab:red', alpha=1.0, lw=2, label='SBR')
    axs.set_ylabel(r'Density [$cm^{-3}$]', fontsize=22)
    axs.set_xlabel(r'Time [UTC]', fontsize=22)

    if lfr_den: 
        qtn_mask = lfr_den.electron_density.data < 0
        lfrden = lfr_den.electron_density.data
        lfrden[qtn_mask] = np.nan
        axs.plot(lfr_den.Epoch.data, lfr_den.electron_density.data, color='tab:orange', label = 'LFR QTN')

    axs.legend(ncols=4, fontsize=18, frameon=False)
    axs.set_ylim([0,4000])
    axs.set_xlim([sbr_time[0], sbr_time[-1]])
    


    plt.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.1)

    plt.show()


if __name__ == "__main__":
    # trange = ['2024-12-24T09:59:59', '2024-12-24T10:15:00']
    # trange = ['2020-01-26T07:00:00', '2020-01-26T07:30:00']
    trange = ['2022-02-25T15:00:00', '2022-02-25T19:00:00']
    credentials = fn.load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    creds = None

    # get the moments.
    mom_data = fn.init_psp_moms(trange, CREDENTIALS=creds, CLIP=True)
    
    # get the biMax fits
    cdfdata = cdflib.cdf_to_xarray('/home/michael/Research/GDF/biMax_Fits/spp_swp_spi_sf00_fits_2024-12-24_v00.cdf', to_datetime=True)
    fit_sel = cdfdata.sel(Epoch=slice(mom_data.Epoch.data[0] - np.timedelta64(1,'s'), mom_data.Epoch.data[-1]))

    # Load the pklfile for SBR.
    # sbr1 = read_pickle('/home/michael/Downloads/vdf_rec_data_2024-12-24_to_100')
    # sbr2 = read_pickle('/home/michael/Downloads/vdf_rec_data_2024-12-24_to_300')
    sbr = read_pickle('/home/michael/Downloads/vdf_rec_data_2020-01-26_to_257')

    # sbr = {**sbr1, **sbr2}

    # Get the LFR
    # sqtn = cdflib.cdf_to_xarray('/home/michael/Downloads/psp_fld_l3_sqtn_rfs_V1V2_20241224_v2.0.cdf', to_datetime=True)
    sqtn = cdflib.cdf_to_xarray('/home/michael/Downloads/psp_fld_l3_sqtn_rfs_V1V2_20200126_v1.0.cdf', to_datetime=True)

    sqtn_sel = sqtn.sel(Epoch=slice(mom_data.Epoch.data[0] - np.timedelta64(1,'s'), mom_data.Epoch.data[-1]))
    