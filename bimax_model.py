import numpy as np
import cdflib
import matplotlib.pyplot as plt

import gdf.src.functions as fn
import astropy.constants as c
import astropy.units as u

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

    # Rotate vel into field aligned coordinates
    u_para_1, u_perp1_1, u_perp2_1 = fn.rotate_vector_field_aligned(*np.array(vel), *fn.field_aligned_coordinates(np.asarray(b_inst)))

    uperp_1 = np.sqrt(u_perp1_1**2 + u_perp2_1**2)

    # Define the centroid of the beam
    u_para_2, u_perp1_2, u_perp2_2 = np.array([u_para_1 + vdrift, u_perp1_1, u_perp2_1])

    print(u_para_1, u_para_2)

    uperp_2 = np.sqrt(u_perp1_2**2 + u_perp2_2**2)

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

if __name__ == '__main__':
    cdfdata = cdflib.cdf_to_xarray('/home/michael/Research/GDF/biMax_Fits/spp_swp_spi_sf00_fits_2024-12-24_v00.cdf', to_datetime=True)

    tidx = np.argmin(np.abs(cdfdata.Epoch.data - np.datetime64('2024-12-24T10:00:01')))

    vpara = np.linspace(-1000,0,101)
    vperp = np.linspace(-1000,1000,101)

    XX, YY = np.meshgrid(vpara, vperp, indexing='ij')
    grids = np.vstack([XX.flatten(), YY.flatten()]).T

    f_bimax = get_model_params(cdfdata, tidx, grids)

    u_para, u_perp1, u_perp2 = fn.rotate_vector_field_aligned(*np.array(cdfdata.vcm.data[tidx]), *fn.field_aligned_coordinates(np.asarray(cdfdata.B_inst.data[tidx])))

    fig, ax = plt.subplots(layout='constrained')
    ax1 = ax.tricontourf(grids[:,1], -grids[:,0] + u_para, np.log10(f_bimax + 1) - 30, levels=np.linspace(-22, -18.5, 8))
    plt.colorbar(ax1)
