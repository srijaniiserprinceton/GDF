# init_gdf.py

# Define the time window of interest
# TRANGE = ['2020-01-26T14:28:00', '2020-01-26T20:30:59']
# TRANGE = ['2020-01-29T18:10:02', '2020-01-29T19:30:59']
# TRANGE = ['2022-02-25T15:55:00', '2022-02-25T15:59:00']
# TRANGE = ['2024-03-30T12:12:00', '2024-03-30T17:30:59']
# TRANGE = ['2018-11-07T03:30:00', '2018-11-07T03:55:00']
# TRANGE = ['2024-09-24T12:12:00', '2024-09-24T17:30:59']
# TRANGE = ['2025-06-19T11:25:00', '2025-06-19T11:45:59']

config = {
    'global': {
        'METHOD'          : 'hybrid',
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T20:30:59'], # Define the time range to load in from pyspedas
        'CLIP'            : True,
        'START_INDEX'     : 0,
        'NSTEPS'          : 200,                                              # use None for entire TRANGE interval
        'CREDS_PATH'      : './config.json',                                           # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : False,
        'SAVE_PKL'        : False,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : False,
        'MCMC_WALKERS'    : 6,
        'MCMC_STEPS'      : 200,
    },
    'polcap': {
        'TH'              : 60,
        'LMAX'            : 12,
        'N2D_POLCAP'      : None,
        'P'               : 3,
        'SPLINE_MINCOUNT' : 7,
    },
    'cartesian': {
        'N2D_CART'        : None,
    },
    'hybrid': {
        'LAMBDA'          : 1e-1,
    }
}