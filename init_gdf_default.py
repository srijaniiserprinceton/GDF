# init_gdf_default.py
# This is a template for primary initization file.
# 'global': parameters needed for any method of reconstruction
# 'polcap', 'cartesian', 'hybrid' are parameters specific to particular methods of reconstruction

config = {
    'global': {
        'METHOD'          : 'cartesian',
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T20:30:59'],
        'CLIP'            : True,
        'START_INDEX'     : 0,
        'NSTEPS'          : 1,
        'CREDS_PATH'      : None,
        'COUNT_THRESHOLD' : 2,
        'SAVE_FIGS'       : True,
        'SAVE_PKL'        : True,
        'MIN_METHOD'      : 'L-BFGS-B',
        'NPTS_SUPER'      : 49,
        'MCMC'            : False,
        'MCMC_WALKERS'    : 6,
        'MCMC_STEPS'      : 200,
    },
    'polcap': {
        'TH'              : 30,
        'LMAX'            : 12,
        'N2D_POLCAP'      : None,
        'P'               : 3,
        'SPLINE_MINCOUNT' : 7,
    },
    'cartesian': {
        'N2D_CART'        : 16,
    },
    'hybrid': {
        'LAMBDA'          : 1e-1,
    }
}