# init_gdf.py

# Define the time window of interest

config = {
    'global': {
        'METHOD'          : 'hybrid',
        'TRANGE'          : ['2025-03-23T00:00:00', '2025-03-23T04:00:00'],
        'SPC_FIT'         : None,
        'SYNTHDATA_FILE'  : None,                                           # Path to a data file containing synthetic observation
        'CLIP'            : True,
        'RESAMPLE'        : None,
        'START_INDEX'     : 0,
        'NSTEPS'          : None,                                              # use None for entire TRANGE interval
        'INDICES'         : None,
        'CREDS_PATH'      : './config.json',                                  # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : False,
        'SAVE_PKL'        : True,
        'SAVE_SUPRES'     : True,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : False,
        'MCMC_WALKERS'    : 6,
        'MCMC_STEPS'      : 200,
        'PARALLELIZED'    : True,
        'N_CORES'         : 4,
    },
    'polcap': {
        'TH'              : None,
        'LMAX'            : 12,
        'N2D_POLCAP'      : None,
        'P'               : 3,
        'SPLINE_MINCOUNT' : 7,
    },
    'cartesian': {
        'N2D_CART'        : 20,
        'N2D_CART_MAX'    : 50,
    },
    'hybrid': {
        'LAMBDA'          : 2.0,
    },
    'quadrature': {
        'NQ_V'             : 2,
        'NQ_T'             : 2,
        'NQ_P'             : 2,
    }
}