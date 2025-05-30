import sys, importlib
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt; plt.ion()

from gdf.src import hybrid_ubulk
from gdf.src import functions as fn
from gdf.src import misc_funcs as misc_fn

N = 12

# config file
config_file = sys.argv[1]

# importing the config file
config = importlib.import_module(config_file)
gvdf_tstamp = hybrid_ubulk.run(config)

# extracting the convex hull boundary
points = np.vstack([gvdf_tstamp.v_para_all, gvdf_tstamp.v_perp_all]).T
tri = Delaunay(points)
idx = np.unique(tri.convex_hull)
points_idx = np.flip(points[idx], axis=1)
angles = np.arctan2(points_idx[:,0], points_idx[:,1]- np.mean(points_idx[:,1]))
sortidx = np.argsort(angles)

# plotting the points and the boundary
plt.figure()
plt.scatter(points_idx[:,0], points_idx[:,1], color='red')
plt.plot(points_idx[:,0][sortidx], points_idx[:,1][sortidx], 'k')

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
        [G,H,V,K,XYP,XY,A] = self.eng.localization2D(XY, N, [], 'RS', [], [], np.array([[10.,10.]]), XYP, nargout=7)
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
Slep2D.gen_Slep_basis(points_idx[sortidx], np.double(N), np.array([xx.flatten(), yy.flatten()]).T)

# plotting the basis functions inside the domain
fig, ax = plt.subplots(2, 10, figsize=(15,3), sharex=True, sharey=True)
for i in range(N):
    row, col = i // 10, i % 10
    ax[row,col].pcolormesh(xx, yy, np.reshape(Slep2D.H[:,i], (49,49), 'C'),
                           cmap='seismic', rasterized=True)
    ax[row,col].plot(points_idx[:,0][sortidx], points_idx[:,1][sortidx], 'k')
    ax[row,col].set_aspect('equal')
    ax[row,col].set_title(r'$\alpha_{%i}$ = '%i + f'{Slep2D.V[i]:.3e}', fontsize=10)

plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.98, wspace=0.02, hspace=0.02)

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
vdfrec = coeffs @ Slep2D.G[:,:N2D].T

# plotting the reconstructed VDF
plt.figure(figsize=(8,6), layout='constrained')
plt.contourf(xx, yy, np.reshape(vdfrec, (49,49), 'C'), levels=np.linspace(0,6.0,20), cmap='plasma')
plt.colorbar()
plt.scatter(eval_gridx, eval_gridy, color='k', s=1)
plt.gca().set_aspect('equal')