import numpy as np
import cdflib 
import xarray as xr
import functions as fn
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    trange = ['2024-12-24T00:00:00', '2024-12-24T12:00:00']
    credentials = fn.load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]

    # get the moments.
    mom_data = fn.init_psp_moms(trange, CREDENTIALS=creds, CLIP=False)
    
    # get the biMax fits
    cdfdata = cdflib.cdf_to_xarray('/home/michael/Research/GDF/biMax_Fits/spp_swp_spi_sf00_fits_2024-12-24_v00.cdf', to_datetime=True)



    