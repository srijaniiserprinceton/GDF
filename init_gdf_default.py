# init_gdf_default.py
# This is a template for primary initization file.
# 'global': parameters needed for any method of reconstruction
# 'polcap', 'cartesian', 'hybrid' are parameters specific to particular methods of reconstruction

config = {
    'global': {                                                             #--------GLOBAL PARAMETERS FOR GDF-----------#
        'METHOD'          : 'hybrid',                                       # choose a method. Use 'hybrid' if unsure
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T14:30:00'], # Define the time range to load in from pyspedas
        'SYNTHDATA_FILE'  : None,                                           # Path to a data file containing synthetic observation
        'CLIP'            : True,                                           # If you want to clip the loaded day's data to the specified TRANGE
        'START_INDEX'     : 0,                                              # Starting index with respect to the first timestamp in TRANGE
        'NSTEPS'          : 1,                                              # use None for entire TRANGE interval
        'CREDS_PATH'      : './config.json',                                # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 2,                                              # the minimum counts per grid point to be consider in the reconstruction
        'SAVE_FIGS'       : True,                                           # flag for saving final figures. Default extension is .png
        'SAVE_PKL'        : False,                                          # flag for saving the reconstructed moments for the NSTEPS reconstruction
        'SAVE_SUPRES'     : False,                                          # if you want to save the super-resolved GDF and the corresponding grids
        'MIN_METHOD'      : 'L-BFGS-B',                                     # minimization method to find gyro-centroid
        'NPTS_SUPER'      : 49,                                             # the number of grids in the super-resolved GDF. Taken scalar or a tuple
        'MCMC'            : False,                                          # flag for using MCMC based gyro-centroid refinement
        'MCMC_WALKERS'    : 6,                                              # number of inpendent walkers to be used in MCMC
        'MCMC_STEPS'      : 200,                                            # number of steps each walker will take
    },
    'polcap': {                                                             #-----------POLAR CAP METHOD PARAMETERS---------#
        'TH'              : None,                                           # the one-sided angle for the polar cap. None = use adaptive calculation
        'LMAX'            : 12,                                             # maximum angular degree for polar cap Slepians, instrument-based choice
        'N2D_POLCAP'      : None,                                           # total number of 1D polcap cap Slepian functions to use. None = Shannon
        'P'               : 3,                                              # order for the piecewise B-spline. 
        'SPLINE_MINCOUNT' : 7,                                              # minimum number of gridpoints for each knot. If lower, then merge knots
    },
    'cartesian': {                                                          #------------CARTESIAN METHOD PARAMETERS---------#
        'N2D_CART'        : None,                                           # default choice for the number of Cartesian Slepian functions.
        'N2D_CART_MAX'    : 100,                                            # maximum value for N2D_CART. Useful for very high calculated kmax
    },
    'hybrid': {                                                             #--------------HYBRID METHOD PARAMETERS-----------#
        'LAMBDA'          : None,                                           # default choice for similarity parameters. None = use L-curve method
    },
    'quadrature': {                                                         #--------------GAUSS-LEGENDRE QUADRATURE----------# [not used currently]
        'NQ_V'             : 2,                                             # number of nodes in velocity
        'NQ_T'             : 2,                                             # number of nodes in elevation (theta)
        'NQ_P'             : 2,                                             # number of nodes in azimuth (phi)
    }
}