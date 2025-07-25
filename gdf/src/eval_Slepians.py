import sys
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
NAX = np.newaxis

from gdf.src import functions as fn
# from gdf.src import coordinate_frame_functions as coor_fn
from gdf.src import basis_funcs as basis_fn

class Slep_transverse:
    def __init__(self):
        """
        Class that contains miscellaneous parameters for the polar cap Slepians.

        * C    : The Slepian tapers (coefficients for SH to generate the Slepian functions)
        * V    : The Slepian concentrations --- eigenvalues of the localization problem.
        * norm : The normalization constants for each Slepian functions, used in the D matrix.
        * eng  : The Matlab engine used to run the .mat codes from inside Python.
        """
        self.slep_dir = fn.read_config()[0]  
        self.C = None       # Gives us the tapers for 1D Legendre Polynomials
        self.V = None       # Concentration coefficient
        self.norm = None    # The norm for the Slepian function

        #--Starting Matlab engine to get the Slepian functions---#
        import matlab.engine as matlab
        self.eng = matlab.start_matlab()
        s = self.eng.genpath(self.slep_dir)
        self.eng.addpath(s, nargout=0)

        # performing setenv for IFILES
        IFILES_PATH = f'{self.slep_dir}/IFILES'
        self.eng.setenv("IFILES", IFILES_PATH, nargout=0)
    
    def gen_Slep_tapers(self, TH, Lmax, m=0, nth=180):
        """
        Generates the Slepian tapers for a given polar cap extent and maximum angular degree resolution.

        Parameters
        ----------
        TH : float
            The polar cap extent in degrees.
        
        Lmax : int
            The maximum angular degree to be used for generating the Slepian functions.

        m : int
            The azimuthal order of the 2D polar cap Slepians. Here, we are interested only 
            in the axisymmetric functions and hence choose m = 0 as default. This will not change
            as long as we are interested in gyrotropic functions.

        nth : int
            The total number of grid points to be used in the Matlab code for the localization problem.
            180 seems to be a reasonably good number from trial and error for our purposes.
        """
        # converting to double type for Matlab code to work
        TH, Lmax, m = np.double(TH), np.double(Lmax), np.double(m)
        #---------------------------------------------------------#
        [E,V,N,th,C] = self.eng.sdwcap(TH, Lmax, m, nth, [], 1, nargout=5)
        self.C = C
        self.V = np.asarray(V).squeeze()
        self.Lmax = Lmax

    def gen_Slep_basis(self, theta_grid):
        r"""
        Generates the Slepian basis functions :math:`S_{\alpha}(\theta)` using the 
        Spherical Harmonics' localization coefficients computed at the initia;ization of the
        `Slep_transverse` class instance.
        """
        [G, th] = self.eng.pl2th(self.C, theta_grid, np.double(1), nargout=2)
        G, th = np.asarray(G).squeeze(), np.asarray(th).squeeze()

        self.G = G
        self.th = th

    def gen_Slep_norms(self):
        r"""
        Generates the normalization constants for each Slepian basis function which is needed
        for generating the D (regularization) matrix. 

        .. math::
        \mathcal{N}_{\alpha} = \int S_{\alpha}(\theta) \, S_{\alpha}(\theta) \, \sin(\theta) \, d\theta
        """
        # creating the theta grid to evaluate the norms
        theta_arr = np.linspace(0, 180, 360)
        S_alpha_theta = basis_fn.get_Slepians_scipy(self.C, theta_arr, self.Lmax)

        # initializing the norm array
        self.norm = np.zeros_like(self.V)

        for i in range(len(self.V)):
            self.norm[i] = fn.norm_eval_theta(S_alpha_theta[i], S_alpha_theta[i], theta_arr)

class Slep_2D_Cartesian:
    def __init__(self):
        """
        Starts the Matlab engine for generating the Cartesian Slepians. This is initialized only
        once in the __init__ of the `gyrovdf` class in `VDF_rec_PSP.py`. NOTE: Further speed-up
        might need us to convert Matlab to Python.
        """
        self.slep_dir = fn.read_config()[0]  

        #--Starting Matlab engine to get the Slepian functions---#
        import matlab.engine as matlab
        self.eng = matlab.start_matlab()
        s = self.eng.genpath(self.slep_dir)
        self.eng.addpath(s, nargout=0)
    
    def gen_Slep_basis(self, XY, N, XYP):
        """
        Generates the Cartesian 2D Slepian basis function for a given boundary and Shannon number, on a 
        grid chosen by the user.

        Parameters
        ----------
        XY : array-like of floats
            The boundary points of shape (Npoints_boundary, 2) where :math:`v_{\perp}` = XY[:,0] and 
            :math:`v_{||}` = XY[:,1].

        N : int
            The Shannon number for the generated Slepian basis. This effectively controls the fineness
            or granularity of the basis functions. A larger N means more basis functions and finer 
            structure resolvability.

        XYP : array-like of floats  
            Array of all the points where we want to evaluate the generated Cartesian Slepian basis. The
            shape is (Npoints_eval, 2) where where :math:`v_{\perp}` = XYP[:,0] and :math:`v_{||}` = XYP[:,1].
        """
        [G,H,V,K,XYP,XY,A] = self.eng.localization2D(XY, N, [], 'GL', [], [], np.array([[10.,10.]]), XYP, nargout=7)
        self.G = np.asarray(G)
        self.H = np.asarray(H)
        self.V = np.asarray(V).squeeze()
        self.A = A
        self.K = K
        self.XY = np.asarray(XY)

if __name__=='__main__':
    # defining the concentration problem to obtain the Slepian tapers
    TH = 75
    Lmax = 12
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

    plt.suptitle('Matlab Slepian generation')

    # plotting the 2D scatter plot colored according to the different Slepian functions
    fig, ax = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    
    for i in range(9):
        row, col = i//3, i%3
        ax[row,col].scatter(fac.vperp[idx, fac.nanmask[idx]], fac.vpara[idx, fac.nanmask[idx]],
                            c=Slep.G[:,i], vmin=-3, vmax=3, cmap='seismic')
        ax[row,col].scatter(-fac.vperp[idx, fac.nanmask[idx]], fac.vpara[idx, fac.nanmask[idx]],
                            c=Slep.G[:,i], vmin=-3, vmax=3, cmap='seismic')
        ax[row,col].set_title(f'$\lambda$ = {Slep.V[i]:.6f}')


    # obtaining the Slepians using Scipy's legendre polynomials
    from scipy.special import eval_legendre
    # generating all the angular degrees we are interested in
    L = np.arange(0,Lmax+1)
    P_scipy = np.asarray([eval_legendre(ell, np.cos(theta_grid * np.pi / 180)) for ell in L])
    # adding the normalization sqrt((2l+1) / 4pi)
    P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]

    G_scipy = P_scipy.T @ np.asarray(Slep.C)

    # remaking the plots
    fig, ax = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    
    # regenerating on a nice uniform theta grid
    Slep.gen_Slep_basis(np.linspace(0, np.pi, 180))
    for i in range(9):
        row, col = i//3, i%3
        ax[row,col].plot(Slep.th * 180 / np.pi, Slep.G[:,i], lw=0.5, color='k')
        ax[row,col].set_title(f'$\lambda$ = {Slep.V[i]:.6f}')
        ax[row,col].axvline(int(TH), ls='dashed', color='k')
        ax[row,col].set_xlim([0,None])

    for i in range(9):
        row, col = i//3, i%3
        ax[row,col].plot(theta_grid, G_scipy[:,i], '.r')

    plt.suptitle('Scipy Slepian generation')