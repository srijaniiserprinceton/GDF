import numpy as np
import cdflib
import xarray as xr
import matplotlib.pyplot as plt; plt.ion()
import astropy.constants as c
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical as c2s
NAX = np.newaxis

import functions as fn

if __name__ == "__main__":
    # trange = ['2020-01-29T00:00:00', '2020-01-29T00:00:00']
    trange = ['2020-01-26T00:00:00', '2020-01-26T23:00:00']
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=None)
    idx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T14:10:42')))
    # idx = 9355
    # idx = 666

    

    time = psp_vdf.unix_time.data
    energy = psp_vdf.energy.data
    theta = psp_vdf.theta.data
    phi = psp_vdf.phi.data
    vdf = psp_vdf.vdf.data

    # masking the zero count bins where we have no constraints
    vdf[vdf == 0] = np.nan
    mask = np.isfinite(vdf)

    m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
    q_p = 1

    velocity = np.sqrt(2 * q_p * energy / m_p)

    # Define the Cartesian Coordinates
    vx = velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    vy = velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    vz = velocity * np.sin(np.radians(theta))

    # file = fn.get_psp_span_mom(trange)
    data = fn.init_psp_moms(trange)

    b_span = data.MAGF_INST.data
    v_span = data.VEL_INST.data

    # # Not shifting into plasma frame to get the correct spherical grid for Slepians
    # ux = vx
    # uy = vy
    # uz = vz

    # Shift into the plasma frame
    ux = vx - v_span[:, 0, NAX, NAX, NAX]
    uy = vy - v_span[:, 1, NAX, NAX, NAX]
    uz = vz - v_span[:, 2, NAX, NAX, NAX]

    # Rotate the plasma frame data into the magnetic field aligned frame.
    v_para, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(ux, uy, uz, *fn.field_aligned_coordinates(b_span)))

    # converting the grid to spherical polar in the field aligned frame
    r, theta, phi = c2s(vperp1[idx], vperp2[idx], v_para[idx])
    r = r.value
    theta = np.degrees(theta.value) + 90
    phi = np.degrees(phi.value)

    # # Rotate the plasma frame data into the magnetic field aligned frame.
    # v_para, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(ux, uy, uz, *fn.field_aligned_coordinates(b_span)))

    # Get the truly gyrotropic VDF
    v_perp = np.sqrt(vperp1**2 + vperp2**2)

    density = data.DENS.data
    avg_den = np.convolve(density, np.ones(10)/10, 'same')      # 1-minute average

    va_vec = ((b_span * u.nT) / (np.sqrt(c.m_p * c.mu0 * avg_den[:,None] * u.cm**(-3)))).to(u.km/u.s).value
    va_mag = np.linalg.norm(va_vec, axis=1)

    # These are for plotting with the tricontourf routine.
    v_para_all = np.concatenate([v_para[idx, mask[idx]], v_para[idx, mask[idx]]])
    v_perp_all = np.concatenate([-v_perp[idx, mask[idx]], v_perp[idx, mask[idx]]])
    vdf_all = np.concatenate([vdf[idx, mask[idx]], vdf[idx, mask[idx]]])

    plt.figure(figsize=(8,48))
    plt.tricontourf(v_perp_all/va_mag[idx], v_para_all/va_mag[idx], np.log10(vdf_all), cmap='inferno')
    plt.scatter(v_perp[idx, mask[idx]]/va_mag[idx], v_para[idx, mask[idx]]/va_mag[idx], marker='.', color='k')
    plt.xlabel(r'$v_{\perp}/v_{a}$')
    plt.ylabel(r'$v_{\parallel}/v_{a}$')

    plt.gca().set_aspect('equal')

    # plotting the grid by coloring according to the coordinate value [for verification]
    fig, ax = plt.subplots(1, 3, figsize=(12,5), sharey=True, layout='constrained')

    v_para = -(v_para - np.linalg.norm(v_span, axis=1)[idx])

    im1 = ax[0].scatter(v_perp[idx, mask[idx]], v_para[idx, mask[idx]], c=r[mask[idx]])
    ax[0].scatter(-v_perp[idx, mask[idx]], v_para[idx, mask[idx]], c=r[mask[idx]])
    ax[0].set_aspect('equal')
    cb1 = fig.colorbar(im1, ax=ax[0], location='bottom', shrink=0.8)
    cb1.set_label('r')

    im2 = ax[1].scatter(v_perp[idx, mask[idx]], v_para[idx, mask[idx]], c=theta[mask[idx]])
    ax[1].scatter(-v_perp[idx, mask[idx]], v_para[idx, mask[idx]], c=theta[mask[idx]])
    ax[1].set_aspect('equal')
    cb2 = fig.colorbar(im2, ax=ax[1], location='bottom', shrink=0.8)
    cb2.set_label(r'$\theta$')

    im3 = ax[2].scatter(v_perp[idx, mask[idx]], v_para[idx, mask[idx]], c=phi[mask[idx]], cmap='berlin')
    ax[2].scatter(-v_perp[idx, mask[idx]], v_para[idx, mask[idx]], c=phi[mask[idx]], cmap='berlin')
    ax[2].set_aspect('equal')
    cb3 = fig.colorbar(im3, ax=ax[2], location='bottom', shrink=0.8)
    cb3.set_label(r'$\phi$')

    [ax[i].set_xlabel(r'$v_{\perp}$', fontsize=12) for i in range(3)]
    ax[0].set_ylabel(r'$v_{\parallel}$', fontsize=12) 