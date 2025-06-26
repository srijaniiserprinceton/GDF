# init_gdf_default.py

# Define the time range to load in from pyspedas
TRANGE = ['2020-01-26T14:28:00', '2020-01-26T15:30:59']

# path to the <.json> file containing credentials
CREDS = None

# Start and End index
START_INDEX = 0
NSTEPS = 3                  # use None for entire TRANGE interval

# Core analysis parameters
TH                = 30
LMAX              = 12
N2D               = 3
P                 = 3
SPLINE_MINCOUNT   = 7
COUNT_MASK        = 2
MU                = 1e-3   # only used for gyroaxis and not super-resolution
CLIP              = True
NPTS_SUPER        = 101
MIN_METHOD        = 'L-BFGS-B'
MCMC              = True
MCMC_WALKERS      = 6
MCMC_STEPS        = 200
SAVE_FIGS         = True
SAVE_PKL          = True