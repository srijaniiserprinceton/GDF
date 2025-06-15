import numpy as np
import cdflib
import matplotlib.pyplot as plt

import gdf.src.functions as fn
import astropy.constants as c
import astropy.units as u

from scipy.interpolate import griddata

plt.ion()

def get_model_params(cdfdata, tidx, grids):
    vpara, vperp = grids[:,0], grids[:,1]

    den_core = cdfdata.np1.data[tidx]
    den_beam = cdfdata.np2.data[tidx]

    # Get the cartesian coordinates for velocity in instrument frame
    vel = cdfdata.vp1.data[tidx]
    vdrift = cdfdata.vdrift.data[tidx]
    
    # Get the parallel and perp temperature
    Trat_1 = cdfdata.Trat1.data[tidx]
    Trat_2 = cdfdata.Trat2.data[tidx]

    Tperp_1 = cdfdata.Tperp1.data[tidx]
    Tperp_2 = cdfdata.Tperp2.data[tidx]

    Tpara_1 = Tperp_1/Trat_1
    Tpara_2 = Tperp_2/Trat_2

    print(Tpara_1, Tperp_1)

    # Convert the temperature to thermal speed
    wperp_1 = np.sqrt((2 * Tperp_1 * u.eV)/c.m_p).to('km/s')
    wpara_1 = np.sqrt((2 * Tpara_1 * u.eV)/c.m_p).to('km/s')

    wperp_2 = np.sqrt((2 * Tperp_2 * u.eV)/c.m_p).to('km/s')
    wpara_2 = np.sqrt((2 * Tpara_2 * u.eV)/c.m_p).to('km/s')

    # get the magnetic field data
    b_inst = cdfdata.B_inst.data[tidx]

    # Since we are making a synthetic model, we will build the VDF centered at the core
    # of the 
    u_para_1 = 0
    u_para_2 = u_para_1 + vdrift

    # Core
    f_core = biMax_model(vpara, vperp, den_core * 1e15, u_para_1, wperp_1.value, wpara_1.value)
    f_beam = biMax_model(vpara, vperp, den_beam * 1e15, u_para_2, wperp_2.value, wpara_2.value)
    f = f_core + f_beam
    return(f)

def biMax_model(v_para, v_perp, den, u_para, w_perp, w_para):
    # all units should be converted to cm!
    const = den/(np.pi**(3/2) * w_perp**2 * w_para)
    exponent = np.exp(-(v_perp**2/w_perp**2) - ((v_para - u_para)**2)/w_para**2)

    print('constant', const, 'exponent', exponent)
    return const*exponent

import xarray as xr
def make_synthetic_cdf(time_array, energy_sort, phi_sort, theta_sort, vdf, bspan, uspan):
    # set the xarray time
    xr_time_array = time_array

    # Zero values.
    eflux_sort = np.zeros_like(vdf)
    count_sort = np.zeros_like(vdf)
    unix_time = np.zeros_like(xr_time_array)

    # Generate the xarray dataArrays for each value we are going to pass
    xr_eflux  = xr.DataArray(eflux_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'eV/cm2-s-ster-eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_energy = xr.DataArray(energy_sort, dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'eV', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.01', 'validmax' : '100000.', 'scale' : 'log'})
    xr_phi    = xr.DataArray(phi_sort,    dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_theta  = xr.DataArray(theta_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'degrees', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'-180', 'validmax' : '360', 'scale' : 'linear'})
    xr_vdf    = xr.DataArray(vdf,         dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'s^3/cm^6', 'fillval' : 'np.array([nan], dtype=float32)', 'validmin':'0.001', 'validmax' : '1e+16', 'scale' : 'log'})
    xr_count  = xr.DataArray(count_sort,  dims = ['time', 'energy_dim', 'theta_dim', 'phi_dim'], coords = dict(time = xr_time_array, energy_dim = np.arange(32), theta_dim = np.arange(8), phi_dim = np.arange(8)), attrs={'units':'integer', 'fillval' : 'np.array([0], dtype=float32)', 'validmin':'0', 'validmax' : '2048', 'scale' : 'linear'})
    xr_unix   = xr.DataArray(unix_time, dims=['time'], coords=dict(time = xr_time_array), attrs={'units' : 'time', 'description':'Unix time'})
    xr_bspan  = xr.DataArray(bspan, dims=['time', 'coor'], coords=dict(time = xr_time_array, coor=np.arange(3)), attrs={'units' : 'None', 'description':'B-unit vector'})
    xr_uspan   = xr.DataArray(uspan, dims=['time', 'coor'], coords=dict(time = xr_time_array, coor=np.arange(3)), attrs={'units' : 'km/s', 'description':'Approximate bulk speed'})

    # Generate the xarray.Dataset
    xr_ds = xr.Dataset({
                        'unix_time' : xr_unix,
                        'eflux'  : xr_eflux,
                        'energy' : xr_energy,
                        'phi' : xr_phi,
                        'theta' : xr_theta,
                        'vdf' : xr_vdf,
                        'counts' : xr_count, 
                        'b_span': xr_bspan,
                        'u_span': xr_uspan,
                       },
                       attrs={'description' : 'Temp SPAN-i data recast into proper format. VDF unit is in s^3/cm^6.'})
    
    return(xr_ds)

if __name__ == '__main__':
    trange = ['2022-02-25T16:56:00', '2022-02-25T16:57:00']
    psp_vdf = fn.init_psp_vdf(trange, CLIP=True)

    cdfdata = cdflib.cdf_to_xarray('/home/michael/Research/GDF/biMax_Fits/spp_swp_spi_sf00_fits_2024-12-24_v00.cdf', to_datetime=True)

    tidx = 100 #np.argmin(np.abs(cdfdata.Epoch.data - np.datetime64('2024-12-24T10:00:01')))

    ND = 50
    NP = 32

    vpara = np.linspace(-500,500,ND)
    vperp = np.linspace(0,1000,ND)
    phi0 = np.linspace(0, 2*np.pi, NP)

    XX, YY = np.meshgrid(vpara, vperp, indexing='ij')
    grids = np.vstack([XX.flatten(), YY.flatten()]).T

    f_bimax = get_model_params(cdfdata, tidx, grids)

    # Get the u_parallel offset.
    #u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*np.array(cdfdata.vcm.data[tidx]), *fn.field_aligned_coordinates(np.asarray(cdfdata.B_inst.data[tidx])))
    Nsteps = 15
    # values = np.linspace(-1.0, 1.0, Nsteps)
    theta = np.linspace(np.pi/2, -np.pi/2, Nsteps)

    vdf_inter = np.zeros((Nsteps, 32, 8, 8))
    b_span = np.zeros((Nsteps, 3))
    u_span = np.zeros((Nsteps, 3))

    for i, val in enumerate(theta[9:10]):
        print(f"{i} has a value of {val}")
        Btemp = np.array([np.cos(val), np.sin(val), 0.2])
        # Btemp = np.array([1,-1,0.2])
        u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*np.array(cdfdata.vcm.data[tidx]), *fn.field_aligned_coordinates(np.asarray(Btemp)))

        # Now we need to define the 3D vdf
        vpara1 = np.repeat(XX, NP).reshape(ND,ND,NP)    # Add in the scalar boost in the parallel direciton!
        vperp1 = YY[:,:,None] * np.cos(phi0)[None, None, :]
        vperp2 = YY[:,:,None] * np.sin(phi0)[None, None, :]

        # Now repeat the f_biMax model valeus
        f_bimax_3D = np.repeat(f_bimax, NP).reshape(-1, NP)

        # Rotate the grids into the instrument frame
        # vxg, vyg, vzg = fn.inverse_rotate_vector_field_aligned(*np.array([vpara1.flatten(), vperp1.flatten(), vperp2.flatten()]), *fn.field_aligned_coordinates(np.asarray(cdfdata.B_inst.data[tidx])))
        vxg, vyg, vzg = fn.inverse_rotate_vector_field_aligned(*np.array([vpara1.flatten(), vperp1.flatten(), vperp2.flatten()]), *fn.field_aligned_coordinates(np.asarray(Btemp)))

        # Add the shift to vxg, vyg, vzg so that we are in the correct instrument frame. 
        uxg, uyg, uzg = cdfdata.vcm.data[tidx]

        vxg -= 500
        vyg += 300
        vzg -= 0

        ux, uy, uz = fn.inverse_rotate_vector_field_aligned(*np.array([u_para, u_perp1, u_perp2]), *fn.field_aligned_coordinates(np.asarray(Btemp)))

        # Define the SPAN-i grids
        vel = 13.85*np.sqrt(psp_vdf.energy.data[0,:,:,:])
        theta = np.radians(psp_vdf.theta.data[0,:,:,:]) + np.pi/2.
        phi = np.radians(psp_vdf.phi.data[0,:,:,:]) #- np.pi

        # Define the new grids 
        vx = vel * np.sin(theta) * np.cos(phi)
        vy = vel * np.sin(theta) * np.sin(phi)
        vz = vel * np.cos(theta)

        # Define the points, values and target points
        points = np.vstack([vxg, vyg, vzg]).T
        points_target = np.vstack([vx.flatten(), vy.flatten(), vz.flatten()]).T

        fvalues = f_bimax_3D.flatten()

        # Interpolate using linear interpolation
        f_interp = griddata(points=points, values=fvalues, xi=points_target, method='linear')
        f_interp = f_interp.reshape(32,8,8)

        vdf_inter[i,:,:,:] = f_interp
        b_span[i,:] = Btemp
        u_span[i,:] = np.array([ux, uy, uz])
        print('completed')


        fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(18,6))
        ax0 = ax[0].tricontourf(grids[:,1], -grids[:,0], np.log10(f_bimax + 1) - 30, levels=np.linspace(-24, -17, 10), cmap='plasma')
        ax[0].set_xlabel(r'$v_{\perp}$')
        ax[0].set_ylabel(r'$v_{\parallel}$')

        vidx, tidx, pidx = np.unravel_index(np.nanargmax(f_interp), (NP,8,8))
        ax[1].scatter(vx[:,tidx,:], vy[:,tidx,:], color='k', marker='.')
        ax1 = ax[1].contourf(vx[:,tidx,:], vy[:,tidx,:], np.log10(f_interp[:,tidx,:] + 1) - 30, levels=np.linspace(-24, -17, 10), cmap='plasma')
        
        ax[2].scatter(vx[:,:,pidx], vz[:,:,pidx], color='k', marker='.')
        ax2 = ax[2].contourf(vx[:,:,pidx], vz[:,:,pidx], np.log10(f_interp[:,:,pidx] + 1) - 30, levels=np.linspace(-24, -17, 10), cmap='plasma')

        ax[1].set_xlabel(r'$v_x$')
        ax[1].set_ylabel(r'$v_y$')
        ax[2].set_xlabel(r'$v_x$')
        ax[2].set_ylabel(r'$v_z$')

        plt.colorbar(ax2)
        plt.savefig(f'/home/michael/Research/GDF/Figures/biMax_figs/{i}.png')
        plt.close()

    ds = make_synthetic_cdf(np.arange(Nsteps), psp_vdf.energy.data[0:Nsteps,:,:,:], 
                            psp_vdf.phi.data[0:Nsteps,:,:,:], psp_vdf.theta.data[0:Nsteps,:,:,:],
                            vdf_inter, b_span, u_span)


