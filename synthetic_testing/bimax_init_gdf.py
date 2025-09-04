# init_gdf.py

# Define the time window of interest
# TRANGE = ['2020-01-26T14:28:00', '2020-01-26T20:30:59']
# TRANGE = ['2020-01-29T18:10:02', '2020-01-29T19:30:59']
# TRANGE = ['2022-02-25T15:55:00', '2022-02-25T15:59:00']
# TRANGE = ['2024-03-30T12:12:00', '2024-03-30T17:30:59']
# TRANGE = ['2018-11-07T03:30:00', '2018-11-07T03:55:00']
# TRANGE = ['2024-09-24T12:12:00', '2024-09-24T17:30:59']

config = {
    'global': {
        'METHOD'          : 'polcap',
        'TRANGE'          : ['2024-12-24T09:59:00', '2024-12-24T10:01:00'], # Define the time range to load in from pyspedas
        # 'SYNTHDATA_FILE'  : None,                                           # Path to a data file containing synthetic observation
        'SYNTHDATA_FILE'        : 'Test_1_synthetic_vdf.cdf',
        # 'SYNTHDATA_FILE'        : 'Test_2_synthetic_vdf.cdf',
        # 'SYNTHDATA_FILE'        : 'Test_3_synthetic_vdf.cdf',
        'CLIP'            : True,
        'START_INDEX'     : 0,
        'NSTEPS'          : None,                                              # use None for entire TRANGE interval
        'CREDS_PATH'      : '/home/michael/Research/GDF/config.json',                                # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : True,
        'SAVE_PKL'        : False,
        'SAVE_SUPRES'     : True,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : False,
        'MCMC_WALKERS'    : 3,
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
        'N2D_CART_MAX'    : 100,
    },
    'hybrid': {
        'LAMBDA'          : None,
    }
}