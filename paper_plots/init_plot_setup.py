# init_gdf.py

config = {
    'global': {
        'METHOD'          : 'polcap',
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T15:28:00'], # Define the time range to load in from pyspedas
        'SYNTHDATA_FILE'  : '/home/michael/Research/GDF/synthetic_testing/paper_models_synthetic_vdf.cdf',                                           # Path to a data file containing synthetic observation
        'CLIP'            : True,
        'START_INDEX'     : 0,
        'NSTEPS'          : 1,                                              # use None for entire TRANGE interval
        'CREDS_PATH'      : None,                                  # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : True,
        'SAVE_PKL'        : False,
        'SAVE_SUPRES'     : False,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : True,
        'MCMC_WALKERS'    : 6,
        'MCMC_STEPS'      : 200,
    },
    'polcap': {
        'TH'              : None,
        'LMAX'            : 12,
        'N2D_POLCAP'      : None,
        'P'               : 3,
        'SPLINE_MINCOUNT' : 7,
    },
    'cartesian': {
        'N2D_CART'        : None,
        'N2D_CART_MAX'    : 50,
    },
    'hybrid': {
        'LAMBDA'          : 0.0,
    },
    'quadrature': {
        'NQ_V'             : 2,
        'NQ_T'             : 2,
        'NQ_P'             : 2,
    }
}