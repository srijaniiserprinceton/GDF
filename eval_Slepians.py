import sys
import numpy as np
import matplotlib.pyplot as plt; plt.ion()

import functions as fn
import coordinate_frame_functions as coor_fn

class Slep_transverse:
    def __init__(self):
        self.slep_dir = fn.read_config()[0]     # '/Users/srijanbharatidas/Documents/Research/Codes/Helioseismology/Slepians/Slepian_Git/'
        self.C = None
        self.V = None

        #--Starting Matlab engine to get the Slepian functions---#
        import matlab.engine as matlab
        self.eng = matlab.start_matlab()
        s = self.eng.genpath(self.slep_dir)
        self.eng.addpath(s, nargout=0)
    
    def gen_Slep_tapers(self, TH, Lmax, m=0, nth=180):
        # converting to double type for Matlab code to work
        TH, Lmax, m = np.double(TH), np.double(Lmax), np.double(m)
        #---------------------------------------------------------#
        [E,V,N,th,C] = self.eng.sdwcap(TH, Lmax, m, nth, [], 1, nargout=5)
        self.C = C
        self.V = np.asarray(V).squeeze()

    def gen_Slep_basis(self, theta_grid):
        [G, th] = self.eng.pl2th(self.C, theta_grid, np.double(1), nargout=2)
        G, th = np.asarray(G).squeeze(), np.asarray(th).squeeze()

        self.G = G
        self.th = th

if __name__=='__main__':
    # defining the concentration problem to obtain the Slepian tapers
    TH = 75
    Lmax = 16
    Slep = Slep_transverse()
    Slep.gen_Slep_tapers(TH, Lmax)

    # generating the Slepian functions on a custom grid without re-generating the tapers
    theta_grid = np.linspace(0, np.pi, 180)
    Slep.gen_Slep_basis(theta_grid)

    # plotting the localization coefficients
    plt.figure()
    plt.plot(Slep.V, '.-')

    fig, ax = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    
    for i in range(9):
        row, col = i//3, i%3
        ax[row,col].plot(Slep.th * 180 / np.pi, Slep.G[:,i], lw=0.5, color='k')
        ax[row,col].set_title(f'$\lambda$ = {Slep.V[i]:.6f}')
        ax[row,col].axvline(int(TH), ls='dashed', color='k')
        ax[row,col].set_xlim([0,None])


    # loading VDF and defining timestamp
    trange = ['2020-01-29T00:00:00', '2020-01-29T00:00:00']
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=None)
    idx = 9355

    # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
    fac = coor_fn.fa_coordinates()
    fac.get_coors(psp_vdf, trange)

    # getting the Slepian functions values on the grids
    theta_grid = fac.theta_fa[idx, fac.nanmask[idx]]

    # obtaining the Slepian functions on this theta grid
    Slep.gen_Slep_basis(theta_grid * np.pi / 180)
    
    for i in range(9):
        row, col = i//3, i%3
        ax[row,col].plot(Slep.th * 180 / np.pi, Slep.G[:,i], '.r')

    # plotting the 2D scatter plot colored according to the different Slepian functions
    fig, ax = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    
    for i in range(9):
        row, col = i//3, i%3
        ax[row,col].scatter(fac.vperp[idx, fac.nanmask[idx]], fac.vpara[idx, fac.nanmask[idx]],
                            c=Slep.G[:,i], vmin=-3, vmax=3, cmap='seismic')
        ax[row,col].scatter(-fac.vperp[idx, fac.nanmask[idx]], fac.vpara[idx, fac.nanmask[idx]],
                            c=Slep.G[:,i], vmin=-3, vmax=3, cmap='seismic')
        ax[row,col].set_title(f'$\lambda$ = {Slep.V[i]:.6f}')