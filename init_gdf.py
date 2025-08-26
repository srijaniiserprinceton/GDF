# init_gdf.py

# Define the time window of interest
# TRANGE = ['2020-01-26T14:28:00', '2020-01-26T20:30:59']
# TRANGE = ['2020-01-29T18:10:02', '2020-01-29T19:30:59']
# TRANGE = ['2022-02-25T15:55:00', '2022-02-25T15:59:00']
# TRANGE = ['2024-03-30T12:12:00', '2024-03-30T17:30:59']
# TRANGE = ['2018-11-07T03:30:00', '2018-11-07T03:55:00']
# TRANGE = ['2024-09-24T12:12:00', '2024-09-24T17:30:59']
# TRANGE = ['2025-06-19T11:25:00', '2025-06-19T11:45:59']
# TRANGE = ['2025-03-23T05:32:40', '2025-03-23T05:33:00']
# TRANGE = ['2025-03-22T22:30:00', '2025-03-22T23:59:59']
# TRANGE = ['2025-03-23T00:30:00', '2025-03-23T07:30:00']
# TRANGE = ['2025-03-23T01:29:00', '2025-03-23T01:29:45']
# TRANGE = ['2025-03-23T01:22:30', '2025-03-23T01:24:00']
# TRANGE = ['2025-03-23T01:19:00', '2025-03-23T01:20:00']
# TRANGE = ['2025-03-23T00:30:00', '2025-03-23T00:31:15']

config = {
    'global': {
        'METHOD'          : 'hybrid',
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T20:30:59'], # Define the time range to load in from pyspedas
        'SYNTHDATA_FILE'  : None,                                           # Path to a data file containing synthetic observation
        'CLIP'            : True,
        'START_INDEX'     : 0,
        'NSTEPS'          : 1,                                              # use None for entire TRANGE interval
        'CREDS_PATH'      : './config.json',                                # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : True,
        'SAVE_PKL'        : True,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : True,
        'MCMC_WALKERS'    : 8,
        'MCMC_STEPS'      : 2000,
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
        'N2D_CART_MAX'    : 100,
    },
    'hybrid': {
        'LAMBDA'          : None,
    },
    'quadrature': {
        'NQ_V'             : 2,
        'NQ_T'             : 2,
        'NQ_P'             : 2,
    }
}