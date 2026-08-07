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
        # 'TRANGE'          : ['2021-01-18T18:00:00', '2021-01-18T18:15:00'], # Jia 2021
        # 'TRANGE'          : ['2021-04-27T11:32:00','2021-04-27T11:36:00'],
        # 'TRANGE'          : ['2025-03-23T00:00:00', '2025-03-23T04:00:00'], # Encounter 23 Fast Stream
        # 'TRANGE'          : ['2025-03-23T03:05:00', '2025-03-23T03:12:00'],
        # 'TRANGE'          : ['2023-09-26T06:00:00', '2023-09-26T12:00:00'], # Define the time range to load in from pyspedas
        # 'TRANGE'          : ['2021-04-26T17:39:00', '2021-04-26T17:45:00'],   # Jia E8 SB 84
        # 'TRANGE'          : ['2020-06-05T17:15:00', '2020-06-05T17:20:00'],
        # 'TRANGE'          : ['2019-04-06T04:50:00', '2019-04-06T05:20:00'],
        # 'TRANGE'          : ['2021-01-14T08:55:25', '2021-01-14T09:55:25'],
        # 'TRANGE'          : ['2024-10-03T09:39:59', '2024-10-03T10:02:26'],
        # 'TRANGE'          : ['2021-01-13T03:11:35', '2021-01-13T03:27:54'],
        # 'TRANGE'          : ['2021-01-15T11:11:19', '2021-01-15T11:13:00'],
        # 'TRANGE'          : ['2023-09-27T00:00:00', '2023-09-27T23:59:59'],
        # 'TRANGE'          : ['2020-01-29T18:09:30', '2020-01-29T18:10:30'],
        'TRANGE'            : ['2025-03-21T17:50:00', '2025-03-21T18:20:00'],
        # 'TRANGE'          : ['2025-06-19T11:00:00', '2025-06-19T11:30:00'],
        # 'TRANGE'          : ['2025-03-23T01:00:00', '2025-03-23T01:10:00'],
        'SPC_FIT'         : None,
        'SYNTHDATA_FILE'  : None,                                           # Path to a data file containing synthetic observation
        'CLIP'            : True,
        'RESAMPLE'        : None,
        'START_INDEX'     : 0,
        'NSTEPS'          : None,                                              # use None for entire TRANGE interval
        # 'INDICES'         : [7794, 10592, 11079, 16734, 21178, 21546, 22025, 
        #                      23483, 27284, 28041, 30334, 30544, 30892, 32673, 
        #                      34117, 34804, 38813, 38817, 40614, 40876, 41062, 
        #                      41496, 47077, 48937, 49356],
        'INDICES' : None, 
        'CREDS_PATH'      : './config.json',                                  # path to the <.json> file containing credentials
        'COUNT_THRESHOLD' : 3,
        'SAVE_FIGS'       : True,
        'SAVE_PKL'        : True,
        'SAVE_SUPRES'     : True,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : False,
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
        'LAMBDA'          : None,
    },
    'quadrature': {
        'NQ_V'             : 2,
        'NQ_T'             : 2,
        'NQ_P'             : 2,
    }
}