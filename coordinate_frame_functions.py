import numpy as np
from astropy.coordinates import cartesian_to_spherical as c2s
import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt; plt.ion()
NAX = np.newaxis

import functions as fn

class fa_coordinates:
    def __init__(self):
        self.tidx = None
        self.vpara, self.vperp1, self.vperp2, self.vperp = None, None, None, None
        self.r_fa, self.theta_fa, self.phi_fa = None, None, None
        self.bspan = None
        self.velocity = None
        self.nanmask = None

    def get_coors(self, vdf_dict, trange, count_mask=0, plasma_frame=True, TH=75, CREDENTIALS=None, CLIP=False):
        self.__init__()

        time = vdf_dict.unix_time.data
        energy = vdf_dict.energy.data
        theta = vdf_dict.theta.data
        phi = vdf_dict.phi.data
        vdf = vdf_dict.vdf.data
        count = vdf_dict.counts.data

        # masking the zero count bins where we have no constraints
        vdf[count <= count_mask] = np.nan
        vdf[vdf == 0] = np.nan
        self.nanmask = np.isfinite(vdf)

        m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
        q_p = 1

        self.velocity = np.sqrt(2 * q_p * energy / m_p)

        # Define the Cartesian Coordinates
        vx = self.velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        vy = self.velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        vz = self.velocity * np.sin(np.radians(theta))

        # filemoms = fn.get_psp_span_mom(trange, CREDENTIALS=CREDENTIALS)
        # print(filemoms)
        data = fn.init_psp_moms(trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

        # obtaining the mangnetic field and v_bulk measured
        self.b_span = data.MAGF_INST.data
        v_span = data.VEL_INST.data
        
        # Shift into the plasma frame
        ux = vx - v_span[:, 0, NAX, NAX, NAX]
        uy = vy - v_span[:, 1, NAX, NAX, NAX]
        uz = vz - v_span[:, 2, NAX, NAX, NAX]
        # uy = vy - 24.95
        # uz = vz - 68.98
        

        # Rotate the plasma frame data into the magnetic field aligned frame.
        vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(ux, uy, uz,
                                                                        *fn.field_aligned_coordinates(self.b_span)))
        self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
        self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

        # Boosting the vparallel
        masked_vpara, masked_vperp = np.ma.masked_array(self.vpara, self.nanmask),\
                                     np.ma.masked_array(self.vperp, self.nanmask)
        # self.max_r = np.nanmax(self.vperp/np.tan(np.radians(TH)) - np.abs(self.vpara), axis=(1,2,3))
        self.max_r = np.nanmax(masked_vperp/np.tan(np.radians(TH)) - np.abs(masked_vpara), axis=(1,2,3))
        self.vpara -= self.max_r[:, NAX, NAX, NAX]

        # converting the grid to spherical polar in the field aligned frame
        r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
        self.r_fa = r.value
        self.theta_fa = np.degrees(theta.value) + 90
        self.phi_fa = np.degrees(phi.value)


if __name__=='__main__':
    trange = ['2020-01-26T00:00:00', '2020-01-26T23:00:00']
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=None)
    #idx = 9355
    idx = np.argmin(np.abs(psp_vdf.time.data - np.datetime64('2020-01-26T14:10:42')))

    fac = fa_coordinates()
    fac.get_coors(psp_vdf, trange)

    filemoms = fn.get_psp_span_mom(trange)
    data = fn.init_psp_moms(filemoms[0])
    density = data.DENS.data
    avg_den = np.convolve(density, np.ones(10)/10, 'same')      # 1-minute average

    va_vec = ((fac.b_span * u.nT) / (np.sqrt(c.m_p * c.mu0 * avg_den[:,None] * u.cm**(-3)))).to(u.km/u.s).value
    va_mag = np.linalg.norm(va_vec, axis=1)

    # These are for plotting with the tricontourf routine
    mask = fac.nanmask
    v_para_all = np.concatenate([fac.vpara[idx, mask[idx]], fac.vpara[idx, mask[idx]]])
    v_perp_all = np.concatenate([-fac.vperp[idx, mask[idx]], fac.vperp[idx, mask[idx]]])
    vdf_all = np.concatenate([psp_vdf.vdf.data[idx, mask[idx]], psp_vdf.vdf.data[idx, mask[idx]]])

    plt.figure(figsize=(8,4))
    plt.tricontourf(v_perp_all/va_mag[idx], v_para_all/va_mag[idx], np.log10(vdf_all), cmap='cool')
    plt.xlabel(r'$v_{\perp}/v_{a}$')
    plt.ylabel(r'$v_{\parallel}/v_{a}$')

    plt.gca().set_aspect('equal')

    # plotting the grid by coloring according to the coordinate value [for verification]
    fig, ax = plt.subplots(1, 3, figsize=(12,3), sharey=True)

    ax[0].scatter(fac.vperp[idx, mask[idx]], fac.vpara[idx, mask[idx]], c=fac.r_fa[idx, mask[idx]])
    ax[0].scatter(-fac.vperp[idx, mask[idx]], fac.vpara[idx, mask[idx]], c=fac.r_fa[idx, mask[idx]])
    ax[0].set_aspect('equal')

    ax[1].scatter(fac.vperp[idx, mask[idx]], fac.vpara[idx, mask[idx]], c=fac.theta_fa[idx, mask[idx]])
    ax[1].scatter(-fac.vperp[idx, mask[idx]], fac.vpara[idx, mask[idx]], c=fac.theta_fa[idx, mask[idx]])
    ax[1].set_aspect('equal')

    ax[2].scatter(fac.vperp[idx, mask[idx]], fac.vpara[idx, mask[idx]], c=fac.phi_fa[idx, mask[idx]])
    ax[2].scatter(-fac.vperp[idx, mask[idx]], fac.vpara[idx, mask[idx]], c=fac.phi_fa[idx, mask[idx]])
    ax[2].set_aspect('equal')