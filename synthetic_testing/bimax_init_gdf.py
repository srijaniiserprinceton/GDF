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
        'METHOD'          : 'hybrid',
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T20:30:59'],
        # 'FILENAME'        : 'bimax_vdf_500_bhat_clockwise_field_rotation_pi_8_to_neg_pi_2_hires.cdf',
        # 'FILENAME'        : 'bimax_vdf_500_250_clockwise_field_rotation_corrected_hires.cdf',
        'FILENAME'        : 'Test_1_synthetic_vdf.cdf',
        # 'FILENAME'        : 'Test_2_synthetic_vdf.cdf',
        # 'FILENAME'        : 'Test_3_synthetic_vdf.cdf',
        'CLIP'            : True,
        'START_INDEX'     : 0,
        'NSTEPS'          : 25,
        'CREDS_PATH'      : None,
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : True,
        'SAVE_PKL'        : False,
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