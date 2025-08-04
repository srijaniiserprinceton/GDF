import sys
import os
import numpy as np
import cdflib
from tqdm import tqdm
import xarray as xr

import matplotlib.pyplot as plt
import gdf.src.functions as fn

import astropy.constants as c
import astropy.units as u

from scipy.interpolate import griddata

# Constants for integration
kB = 1.380649e-23  # J/K
qe = 1.602176634e-19  # Elementary charge [C]
mass_p_kg = 1.6726219e-27 

plt.ion()

def biMax_model(v_para, v_perp, den, u_para, w_perp, w_para):
    '''
    This is a simple biMaxwellian model. See Verscharen et al. 2019.
    '''
    # all units should be converted to cm!
    const    = den/(np.pi**(3/2) * w_perp**2 * w_para)
    exponent = np.exp(-(v_perp**2/w_perp**2) - ((v_para - u_para)**2)/w_para**2)

    return const*exponent

def biMax_core_beam(v_para_grid, v_perp_grid, den_core, den_beam, vdrift, w_perp_core, w_para_core, w_perp_beam, w_para_beam):
    bimax_core = biMax_model(v_para_grid, v_perp_grid, den_core,      0, w_perp_core, w_para_core)
    bimax_beam = biMax_model(v_para_grid, v_perp_grid, den_beam, vdrift, w_perp_beam, w_para_beam)

    return bimax_core + bimax_beam

def bimax_model_from_JayeFits(cdfdata, grids):
    '''
    This function uses data form Jaye Verniero's PSP ion fits to 
    construct a synthetic ion VDF. 

    Parameters:
    ----------
    cdfdata : xarray data set 
    '''

    vpara, vperp = grids[:,0], grids[:,1]
    # Get the parallel and perp temperature
    Trat_1 = cdfdata.Trat1.data
    Trat_2 = cdfdata.Trat2.data

    Tperp_1 = cdfdata.Tperp1.data
    Tperp_2 = cdfdata.Tperp2.data

    Tpara_1 = Tperp_1/Trat_1
    Tpara_2 = Tperp_2/Trat_2

    # Convert the temperature to thermal speed
    wperp_1 = np.sqrt((2 * Tperp_1 * u.eV)/c.m_p).to('km/s').value
    wpara_1 = np.sqrt((2 * Tpara_1 * u.eV)/c.m_p).to('km/s').value

    wperp_2 = np.sqrt((2 * Tperp_2 * u.eV)/c.m_p).to('km/s').value
    wpara_2 = np.sqrt((2 * Tpara_2 * u.eV)/c.m_p).to('km/s').value

    den1 = cdfdata.np1.data
    den2 = cdfdata.np2.data
    den_tot = den1 + den2

    f = biMax_core_beam(vpara, vperp, cdfdata.np1.data*1e15, cdfdata.np2.data*1e15,
                        cdfdata.vdrift.data, wperp_1, wpara_1, wperp_2, wpara_2)
    
    print(den1, den2, Trat_1, Trat_2, Tperp_1, Tperp_2, cdfdata.vdrift.data)

    return(f*1e-30, den_tot)

def bimax_model_from_values(grids, den1, den2, Trat1, Trat2, Tperp1, Tperp2, vdrift):
    '''
    This function uses data form Jaye Verniero's PSP ion fits to 
    construct a synthetic ion VDF. 

    Parameters:
    ----------
    cdfdata : xarray data set 
    '''

    vpara, vperp = grids[:,0], grids[:,1]
    # Get the parallel and perp temperature
    Trat_1 = Trat1
    Trat_2 = Trat2

    Tperp_1 = Tperp1
    Tperp_2 = Tperp2

    Tpara_1 = Tperp_1/Trat_1
    Tpara_2 = Tperp_2/Trat_2

    # Convert the temperature to thermal speed
    wperp_1 = np.sqrt((2 * Tperp_1 * u.eV)/c.m_p).to('km/s').value
    wpara_1 = np.sqrt((2 * Tpara_1 * u.eV)/c.m_p).to('km/s').value

    wperp_2 = np.sqrt((2 * Tperp_2 * u.eV)/c.m_p).to('km/s').value
    wpara_2 = np.sqrt((2 * Tpara_2 * u.eV)/c.m_p).to('km/s').value

    den_tot = den1 + den2

    f = biMax_core_beam(vpara, vperp, den1*1e15, den2*1e15,
                        vdrift, wperp_1, wpara_1, wperp_2, wpara_2)
    
    return(f*1e-30, den_tot)

def make_3d(ND, NP, vpara_2d, vperp_2d, phi0, vdf):
    # Now we need to define the 3D vdf
    vpara_3D = np.repeat(vpara_2d, NP).reshape(ND,ND,NP)  # Add in the scalar boost in the parallel direciton!
    vperp1_3D = vperp_2d[:,:,None] * np.cos(phi0)[None, None, :]
    vperp2_3D = vperp_2d[:,:,None] * np.sin(phi0)[None, None, :]

    # Now repeat the f_biMax model valeus
    vdf_bimax_3D = np.repeat(vdf, NP).reshape(ND, ND, NP)    

    return vpara_3D, vperp1_3D, vperp2_3D, vdf_bimax_3D

def shift_frame(vpara_3d, vperp1_3d, vperp2_3d, bvec, uvec):
    vxg, vyg, vzg = fn.inverse_rotate_vector_field_aligned(
        *np.array([vpara_3d.flatten(), vperp1_3d.flatten(), vperp2_3d.flatten()]), 
        *fn.field_aligned_coordinates(np.asarray(bvec))
    )

    vxg += uvec[0]
    vyg += uvec[1]
    vzg += uvec[2]

    return(vxg, vyg, vzg)

def generate_inst_grid(vdf_xr):
    # Define the SPAN-i grids
    e_inst = vdf_xr.energy.data[0,:,:,:]
    vel_inst = 13.85*np.sqrt(vdf_xr.energy.data[0,:,:,:])
    theta_inst = np.radians(vdf_xr.theta.data[0,:,:,:])
    phi_inst = np.radians(vdf_xr.phi.data[0,:,:,:])

    # Define the new grids 
    vx_inst = vel_inst * np.cos(theta_inst) * np.cos(phi_inst)
    vy_inst = vel_inst * np.cos(theta_inst) * np.sin(phi_inst)
    vz_inst = vel_inst * np.sin(theta_inst)

    return(e_inst, vel_inst, theta_inst, phi_inst, vx_inst, vy_inst, vz_inst)

def check_density_and_temperatures(ntot, vpara_hr_3d, vperp1_hr_3d, vperp2_hr_3d, f_bimax_3d, mass_kg=1.6726e-27):
    # Calculate grid spacing (assumes uniform spacing in each dimension)
    dvpara = np.mean(np.diff(vpara_hr_3d[:, 0, 0]))
    dvperp = np.mean(np.diff(vperp1_hr_3d[0, :, 0]))

    vperp3d = np.sqrt(vperp1_hr_3d**2 + vperp2_hr_3d**2)

    den_val = 2*np.pi*np.sum(vperp3d[:,:,0]*1e15*f_bimax_3d[:,:,0]*dvpara*dvpara) # volume element in velocity space (m^3/s^3)

    m_p = 1.6726e-24        # g        
    k_b = 1.380649e-16      # erg/K

    T_para = (m_p/k_b)*(2*np.pi*np.sum((vpara_hr_3d[:,:,0] * 1e5)**2 * vperp3d[:,:,0]*1e5 * f_bimax_3d[:,:,0] * dvpara*1e5 * dvperp*1e5)/den_val)
    T_perp = (m_p/(2*k_b))*(2*np.pi*np.sum((vperp3d[:,:,0] * 1e5)**2 * vperp3d[:,:,0]*1e5 * f_bimax_3d[:,:,0] * dvpara*1e5 * dvperp*1e5)/den_val)

    print(f"The calculated density is {den_val} (relative error: {100*np.abs(den_val - ntot)/ntot:.2f}%)")
    print(f"Parallel temperature: {T_para} J")
    print(f"Perpendicular temperature: {T_perp} J")
    
    return den_val, T_para, T_perp


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

def model_params():
    # Step I: Define the parameters needed for the bi-max model
    den1   = 1466.9059591788823 
    den2   = 236.0497816330413 
    Trat1  = 3.078559367462411 
    Trat2  = 0.9010196661059698 
    Tperp1 = 76.73630778076499 
    Tperp2 = 105.61544387731757 
    vdrift = -89.79244089265619

    return(den1, den2, Trat1, Trat2, Tperp1, Tperp2, vdrift)

class synthetic_models:
    def __init__(self, den1, den2, Trat1, Trat2, Tperp1, Tperp2, vdrift):
        # Step I: Define the parameters needed for the bi-max model
        self.den1   = den1 
        self.den2   = den2
        self.Trat1  = Trat1
        self.Trat2  = Trat2
        self.Tperp1 = Tperp1 
        self.Tperp2 = Tperp2
        self.vdrift = vdrift
        
        self.ND = 50
        self.NP = 32

        self.vpara_hr = np.linspace(-600,600,self.ND)
        self.vperp_hr = np.linspace(0,1200,self.ND)
        self.phi0 = np.linspace(0, 2*np.pi, self.NP)

        self.XX, self.YY = np.meshgrid(self.vpara_hr, self.vperp_hr, indexing='ij')
        self.grids_hr = np.vstack([self.XX.flatten(), self.YY.flatten()]).T
        # Generate the bi-Maxwellian
        self.f_bimax, self.ntot = bimax_model_from_values(self.grids_hr, self.den1, self.den2, self.Trat1, 
                                                           self.Trat2, self.Tperp1, self.Tperp2, self.vdrift)
        
        # self.init_psp_inst_grids()

    def define_inst_grids(self, energy_array, theta_array, phi_array):
        self.energy_inst = energy_array[:, np.newaxis, np.newaxis] * np.ones((1, len(theta_array), len(phi_array)))
        self.theta_inst  = theta_array[np.newaxis, : , np.newaxis] * np.ones((len(energy_array), 1, len(phi_array)))
        self.phi_inst    = phi_array[np.newaxis, np.newaxis, :] * np.ones((len(energy_array), len(theta_array), 1))

        self.vel_inst    = 13.85*np.sqrt(self.energy_inst * 1.0)

        # Define the new grids 
        self.vx_inst = self.vel_inst * np.cos(np.radians(self.theta_inst)) * np.cos(np.radians(self.phi_inst))
        self.vy_inst = self.vel_inst * np.cos(np.radians(self.theta_inst)) * np.sin(np.radians(self.phi_inst))
        self.vz_inst = self.vel_inst * np.sin(np.radians(self.theta_inst))

    def init_psp_inst_grids(self):
        # Define the energy array 
        energy_array = np.array([   21.020508,    26.02539,    32.03125,    40.039062,
                                    49.047850,    61.05957,    76.07422,    95.092770,
                                   118.115234,   147.14355,   182.17773,   227.221680,
                                   282.275400,   351.34277,   436.42578,   542.529300,
                                   673.657200,   837.81740,  1041.01560,  1294.262700,
                                  1608.569300,  1999.95120,  2486.42580,  3090.014600,
                                  3841.748000,  4774.65800,  5935.79100,  7378.198000,
                                  9171.948000, 11401.12300, 14171.82600, 17616.186000])
        
        theta_array = np.array([-51.775665, -37.69623, -22.3178, -7.6308002,   
                                7.5497575, 22.527119, 37.425697, 52.47127])
        
        phi_array = np.array([174.375, 163.125, 151.875, 140.625, 129.375, 118.125, 106.875, 95.625])

        self.energy_inst = energy_array[:, np.newaxis, np.newaxis] * np.ones((1, len(theta_array), len(phi_array)))
        self.theta_inst  = theta_array[np.newaxis, : , np.newaxis] * np.ones((len(energy_array), 1, len(phi_array)))
        self.phi_inst    = phi_array[np.newaxis, np.newaxis, :] * np.ones((len(energy_array), len(theta_array), 1))

        self.vel_inst    = 13.85*np.sqrt(self.energy_inst * 1.0)

        # Define the new grids 
        self.vx_inst = self.vel_inst * np.cos(np.radians(self.theta_inst)) * np.cos(np.radians(self.phi_inst))
        self.vy_inst = self.vel_inst * np.cos(np.radians(self.theta_inst)) * np.sin(np.radians(self.phi_inst))
        self.vz_inst = self.vel_inst * np.sin(np.radians(self.theta_inst))

    def make_3d_bimax(self):
        temp_arr = make_3d(self.ND, self.NP, self.XX, self.YY, self.phi0, self.f_bimax)
        self.vpara_hr_3d, self.vperp1_hr_3d, self.vperp2_hr_3d, self.f_bimax_3d = temp_arr
        
        n3d = check_density_and_temperatures(self.ntot, self.vpara_hr_3d, 
                                             self.vperp1_hr_3d, self.vperp2_hr_3d, 
                                             self.f_bimax_3d)
    
    def interpolate_to_inst_grids(self):
        target_points = np.vstack([self.vx_inst.flatten(), self.vy_inst.flatten(), self.vz_inst.flatten()]).T
        self.vdf_inter = np.zeros((self.Nsteps, 32, 8, 8))
        for i in tqdm(range(self.Nsteps)):
            vxg, vyg, vzg = shift_frame(self.vpara_hr_3d, self.vperp1_hr_3d, self.vperp2_hr_3d, self.bvecs[i], self.uvecs[i])

            grids = np.asarray([vxg, vyg, vzg])

            self.f_interp = griddata(points=np.vstack([vxg, vyg, vzg]).T,
                                    values = self.f_bimax_3d.flatten(),
                                    xi=target_points, 
                                    method='linear')
            
            self.vdf_inter[i,:,:,:] = self.f_interp.reshape(32,8,8)
            self.plot_slices(i)
            print(fn.compute_vdf_moments(self.energy_inst[:,0,0], np.radians(90.0 - self.theta_inst[0,:,0]), np.radians(self.phi_inst[0,0,:]), self.vdf_inter[i], mass_p_kg))

        psp_energy = np.array([self.energy_inst for _ in range(self.Nsteps)]) 
        psp_theta  = np.array([self.theta_inst for _ in range(self.Nsteps)])  
        psp_phi    = np.array([self.phi_inst for _ in range(self.Nsteps)])    

        self.ds = make_synthetic_cdf(np.arange(self.Nsteps), psp_energy, psp_phi, psp_theta, self.vdf_inter, self.bvecs, self.uvecs)

    def synthetic_test_1(self, Nsteps = 100):
        self.init_psp_inst_grids()
        self.make_3d_bimax()

        # define the magnetic field vector
        # Generate the set of magnetic field vectors that we are going to use
        theta = np.linspace(np.pi/8, -np.pi/2, Nsteps)

        self.bvecs = np.asarray([np.cos(theta), np.sin(theta), np.repeat(0.2, Nsteps)]).T
        self.uvecs = -500*self.bvecs/np.linalg.norm(self.bvecs, axis=1)[:,None]
        self.Nsteps = Nsteps

        self.tag = 'Test_1'

        self.interpolate_to_inst_grids()

    def synthetic_test_2(self, Nsteps = 100):
        self.init_psp_inst_grids()
        self.make_3d_bimax()

        # define the magnetic field vector
        # Generate the set of magnetic field vectors that we are going to use
        theta = np.linspace(np.pi/2, -np.pi/2, Nsteps)

        self.bvecs = np.asarray([np.cos(theta), np.sin(theta), np.repeat(0.2, Nsteps)]).T
        self.uvecs = np.array([-500, 250, 0]) * np.ones(Nsteps, 3)
        self.Nsteps = Nsteps

        self.tag = 'Test_2'

        self.interpolate_to_inst_grids()

    def define_model_vectors(self, bvec, uvec, TAG = 'Custom'):
        self.make_3d_bimax()

        self.bvecs = bvec
        self.uvecs = uvec
        self.Nsteps = len(bvec)

        self.tag = TAG

    def plot_slices(self, i):
        fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(18,6))
        ax0 = ax[0].tricontourf(self.grids_hr[:,1], -self.grids_hr[:,0], np.log10(self.f_bimax), levels=np.linspace(-24, -17, 10), cmap='plasma')
        ax[0].set_xlabel(r'$v_{\perp}$')
        ax[0].set_ylabel(r'$v_{\parallel}$')
        vidx, tidx, pidx = np.unravel_index(np.nanargmax(self.f_interp), (self.NP,8,8))
        ax[1].scatter(self.vx_inst[:,tidx,:], self.vy_inst[:,tidx,:], color='k', marker='.')
        ax1 = ax[1].contourf(self.vx_inst[:,tidx,:], self.vy_inst[:,tidx,:], np.log10(self.vdf_inter[i,:,tidx,:]), levels=np.linspace(-24, -17, 10), cmap='plasma')
        ax[2].scatter(self.vx_inst[:,:,pidx], self.vz_inst[:,:,pidx], color='k', marker='.')
        ax2 = ax[2].contourf(self.vx_inst[:,:,pidx], self.vz_inst[:,:,pidx], np.log10(self.vdf_inter[i,:,:,pidx]), levels=np.linspace(-24, -17, 10), cmap='plasma')
        ax[1].set_xlabel(r'$v_x$')
        ax[1].set_ylabel(r'$v_y$')
        ax[2].set_xlabel(r'$v_x$')
        ax[2].set_ylabel(r'$v_z$')
        plt.colorbar(ax2)
        plt.savefig(f'/home/michael/Research/GDF/Figures/slices/slice_{self.tag}_{i}.png')
        plt.close()

    def save_cdf(self):
        cdflib.xarray_to_cdf(self.ds, f'{self.tag}_synthetic_vdf.cdf')

if __name__ == '__main__':
    if os.path.exists('./Test_1_synthetic_vdf.cdf'):
        print(f"Synthetic test case 1 exists. Continuing...")
    else:
        test1 = synthetic_models(*model_params()).synthetic_test_1()
        test1.save_cdf()
    
    if os.path.exists('./Test_2_synthetic_vdf.cdf'):
        print(f"Synthetic test case 2 exists. Continuing...")
    else:
        test1 = synthetic_models(*model_params()).synthetic_test_1()
        test1.save_cdf()

    

    
    