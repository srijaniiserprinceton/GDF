import sys, importlib
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.colors as colors
plt.rcParams['font.size'] = 16

# from gdf.src import hybrid_ubulk
from bimax_fit_ubulk import run
from gdf.src import functions as fn
from gdf.src import misc_funcs as misc_fn

N = 25

# config file
config_file = sys.argv[1]

# importing the config file
config = importlib.import_module(config_file)
gvdf_tstamp = run(config)

# extracting the convex hull boundary
points = np.vstack([gvdf_tstamp.v_para_all, gvdf_tstamp.v_perp_all]).T
tri = Delaunay(points)
idx = np.unique(tri.convex_hull)
points_idx = np.flip(points[idx], axis=1)
angles = np.arctan2(points_idx[:,0], points_idx[:,1]- np.mean(points_idx[:,1]))
sortidx = np.argsort(angles)

class Slep_2D:
    def __init__(self):
        self.slep_dir = fn.read_config()[0]  
        self.C = None       # Gives us the tapers for 1D Legendre Polynomials
        self.V = None       # Concentration coefficient
        self.norm = None    # The norm for the Slepian function

        #--Starting Matlab engine to get the Slepian functions---#
        import matlab.engine as matlab
        self.eng = matlab.start_matlab()
        s = self.eng.genpath(self.slep_dir)
        self.eng.addpath(s, nargout=0)
    
    def gen_Slep_basis(self, XY, N, XYP):
        # converting to double type for Matlab code to work
        # XY, N, XYP = np.double(XY), int(N), np.double(XYP)
        #---------------------------------------------------------#
        [G,H,V,K,XYP,XY,A] = self.eng.localization2D(XY, N, [], 'GL', [], [], np.array([[11.,11.]]), XYP, nargout=7)
        self.G = np.asarray(G)
        self.H = np.asarray(H)
        self.V = np.asarray(V).squeeze()
        self.A = A
        self.K = K

# making the grid on which to evaluate the Slepian functions
eval_gridx = np.linspace(points_idx[:,0].min(), points_idx[:,0].max(), 49)
eval_gridy = np.linspace(points_idx[:,1].min(), points_idx[:,1].max(), 49)
xx, yy = np.meshgrid(eval_gridx, eval_gridy, indexing='ij')

Slep2D = Slep_2D()
points_sort = points_idx[sortidx]
points_sort = np.vstack([points_sort, points_sort[0]])
# Slep2D.gen_Slep_basis(points_idx[sortidx], np.double(N), np.array([xx.flatten(), yy.flatten()]).T)
Slep2D.gen_Slep_basis(points_sort, np.double(N), np.array([xx.flatten(), yy.flatten()]).T)


# plotting the basis functions inside the domain
fig, ax = plt.subplots(2, 6, figsize=(7,3.1), sharex=True, sharey=True)
for i in range(12):
    row, col = i // 6, i % 6
    ax[row,col].pcolormesh(xx, yy, np.reshape(Slep2D.H[:,i], (49,49), 'C'),
                           cmap='seismic', rasterized=True)
    ax[row,col].plot(points_idx[:,0][sortidx], points_idx[:,1][sortidx], 'k')
    ax[row,col].set_aspect('equal')
    ax[row,col].set_title(r'$\mathbf{\alpha_{%i}}$='%(i+1) + f'{Slep2D.V[i]:.3f}', fontsize=10)
    ax[row,col].tick_params(axis='both', labelsize=7)
    ax[row,col].tick_params(axis='both', labelsize=7)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.98, wspace=0.02, hspace=0.02)

fig.supxlabel(r'$v_{\perp}$', fontsize=12)
fig.supylabel(r'$v_{\parallel}$', fontsize=12)

tidx = config.START_INDEX
plt.savefig('paper_plots/bimax_Slepian_2D_basis_{tidx}.pdf')

# setting up the inversion process
eval_gridx = np.append(-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vperp_nonan)
eval_gridy = np.append(gvdf_tstamp.vpara_nonan, gvdf_tstamp.vpara_nonan)

# getting the Slepians on the measurement points
Slep2D_ = Slep_2D()
Slep2D_.gen_Slep_basis(points_idx[sortidx], np.double(N), np.array([eval_gridx, eval_gridy]).T)

# clipping off at the Shannon number
N2D = int(np.sum(Slep2D_.V))
Slep2D_.G = Slep2D_.G[:,:N2D]
Slep2D_.H = Slep2D_.H[:,:N2D]

# removing the odd basis functions

# the data we intend to fit to
vdf_data = np.append(gvdf_tstamp.vdfdata, gvdf_tstamp.vdfdata)

# performing the inversion
GTG = Slep2D_.G.T @ Slep2D_.G 
coeffs = np.linalg.inv(GTG) @ Slep2D_.G.T @ vdf_data

# reconstructing
vdfrec = coeffs @ Slep2D.H[:,:N2D].T


frec = np.power(10, vdfrec) * gvdf_tstamp.minval[tidx]
f_data = np.power(10, vdf_data) * gvdf_tstamp.minval[tidx]

cmap = plt.cm.plasma
# lvls = np.linspace(int(np.log10(gvdf_tstamp.minval[tidx]) - 1),
#                    int(np.log10(gvdf_tstamp.maxval[tidx])+1), 10)
lvls = np.linspace(-24, -17, 10)        #np.linspace(-22, -18.5, 8)
norm = colors.BoundaryNorm(lvls, cmap.N)

xmagmax = eval_gridx.max() * 1.12

# plotting the points and the boundary
fig, ax = plt.subplots(1, 2, figsize=(7.5,3.5), sharey=True)
ax[0].plot(points_idx[:,0][sortidx], points_idx[:,1][sortidx], '--k')
ax[0].scatter(eval_gridx, eval_gridy, c=np.log10(f_data), s=50, edgecolor='k', linewidths=0.5, cmap='plasma', norm=norm)
ax[0].set_aspect('equal')
ax[0].set_xlim([-xmagmax, xmagmax])

ax[1].plot(points_idx[:,0][sortidx], points_idx[:,1][sortidx], '--w')
ax1 = ax[1].tricontourf(xx.flatten(), yy.flatten(), np.log10(frec), levels=lvls, cmap='plasma')
# ax[1].scatter(eval_gridx, eval_gridy, c=np.log10(f_data), s=50, edgecolor='k', linewidths=0.5, cmap='plasma', norm=norm)
ax[1].set_aspect('equal')
ax[1].set_xlim([-xmagmax, xmagmax])

fig.supxlabel(r'$v_{\perp}$', fontsize=19)
fig.supylabel(r'$v_{\parallel}$', fontsize=19)

plt.subplots_adjust(top=0.98, bottom=0.18, left=0.14, right=0.98, wspace=0.05, hspace=0.05)
plt.savefig('paper_plots/Bimax_Sleprec_comparison_{tidx}.pdf')