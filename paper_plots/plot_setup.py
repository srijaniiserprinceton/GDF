import numpy as np
import os, sys, importlib
import matplotlib.pyplot as plt; plt.ion()
plt.rcParams['font.size'] = 14

from gdf.src import hybrid_ubulk
from gdf.src import basis_funcs as basis_fn

# loading the config file to obtain the desired gvdf dictionary
current_file_dir = os.path.dirname(__file__)
# expects that the init file will be in the local directory of this script
config_file = 'init_plot_setup'

# importing the config file
config = importlib.import_module(config_file, package=current_file_dir)
gvdf_tstamp = hybrid_ubulk.run(config)

# plotting the Bsplines
sortidx = np.argsort(gvdf_tstamp.super_rfac)
rgrid = gvdf_tstamp.super_rfac[sortidx]
rmin = gvdf_tstamp.rfac_nonan.min()
rmax = gvdf_tstamp.rfac_nonan.max()
B_i_n = gvdf_tstamp.super_B_i_n[:,sortidx]

#------------ plotting the Bspline grid setup --------------#
Bsp_idx = 6
plt.figure()

for i in range(len(B_i_n)):
    plt.plot(rgrid, B_i_n[i], 'k')

plt.plot(rgrid, B_i_n[Bsp_idx], 'r', lw=4)

plt.xlim([rmin * 0.95, rmax * 1.05])
plt.xlabel('Radial grid [km/s]', fontweight='bold')
plt.ylabel('Bspline (normalized units)', fontweight='bold')
plt.grid(True)
plt.tight_layout()

# making the 2D vperp-vpara grid and the corresponding 2D rgrid
NPTS = 1000
x = np.linspace(0, gvdf_tstamp.vperp_nonan.max(), NPTS)
y = np.linspace(gvdf_tstamp.vpara_nonan.min(), gvdf_tstamp.vpara_nonan.max(), NPTS)

xx, yy = np.meshgrid(x, y, indexing='ij')
rr  = np.sqrt(xx**2 + yy**2)

Bspline_i_rr = basis_fn.get_Bsplines_scipy(gvdf_tstamp.knots, 3, rr)

#------------- plotting the Slepian grid setup ---------------#
plt.figure()
plt.pcolormesh(xx, yy, Bspline_i_rr[Bsp_idx].T, cmap='Reds', rasterized=True, alpha=0.6)
# Bspline mask to find which points lie on the represented Bspline
r_Bspline = rgrid[np.where(B_i_n[Bsp_idx] > 0.1)]
rmin_Bspline, rmax_Bspline = r_Bspline.min(), r_Bspline.max()
Bspline_mask = (np.sqrt(gvdf_tstamp.vperp_nonan**2 + gvdf_tstamp.vpara_nonan**2) >= rmin_Bspline) *\
               (np.sqrt(gvdf_tstamp.vperp_nonan**2 + gvdf_tstamp.vpara_nonan**2) <= rmax_Bspline)
plt.plot(gvdf_tstamp.vperp_nonan[Bspline_mask], gvdf_tstamp.vpara_nonan[Bspline_mask], 'ok')
plt.plot(gvdf_tstamp.vperp_nonan[~Bspline_mask], gvdf_tstamp.vpara_nonan[~Bspline_mask], '.k')
plt.ylim([rmin * 0.95, rmax * 1.05])
plt.gca().set_aspect('equal')
plt.xlabel(r'$\mathbf{v}_{\perp}$ [km/s]', fontweight='bold')
plt.ylabel(r'$\mathbf{v}_{||}$ [km/s]', fontweight='bold')
plt.tight_layout()

# obtaining the smooth Slepian functions on a dense regular grid
theta_grid = np.linspace(0, 180, 360)
S_alpha_theta = basis_fn.get_Slepians_scipy(gvdf_tstamp.Slep.C, theta_grid,
                                            gvdf_tstamp.Lmax, gvdf_tstamp.N2D)
plt.figure()
for i in range(gvdf_tstamp.N2D):
    # 9 is the index of the timestamp we are plotting in the plot_init_setup.py file
    plt.plot(gvdf_tstamp.theta_fa[gvdf_tstamp.nanmask[9]][Bspline_mask],
             gvdf_tstamp.S_alpha_n[i][Bspline_mask], 'ok', alpha=0.6)
    plt.plot(theta_grid, S_alpha_theta[i], label=r'$S_{%i}(\theta)$'%(i))

plt.axvline(gvdf_tstamp.TH, color='red', ls='dashed')
plt.xlabel(r'$\mathbf{\theta}$ [in degrees]', fontweight='bold')
plt.ylabel('Slepians (normalized units)', fontweight='bold')
plt.legend()



