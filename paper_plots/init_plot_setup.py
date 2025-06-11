# init_gdf.py

# Define the time range we are running for
TRANGE = ['2022-02-25T15:55:00', '2022-02-25T16:00:00']
# TRANGE = ['2020-01-26T14:28:00', '2020-01-26T15:30:59']

# Get the credentials
CREDS = None       # credential file.

# Start and End index
START_INDEX = 0
NSTEPS = 1

# Core analysis parameters
TH                = 60
LMAX              = 12
N2D               = 3
SPLINE_MINCOUNT   = 7
COUNT_MASK        = 1
MU                = 1e-3
CLIP              = True
NPTS_SUPER        = 101
MCMC              = True
MCMC_WALKERS      = 6
MCMC_STEPS        = 200
MIN_METHOD        = 'L-BFGS-B'
SAVE_FIGS         = False
SAVE_PKL          = False