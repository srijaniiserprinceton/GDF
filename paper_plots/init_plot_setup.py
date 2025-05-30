# init_gdf.py

# Define the time range we are running for
TRANGE = ['2022-02-25T16:56:00', '2022-02-25T17:01:00']

# Get the credentials
CREDS = None       # credential file.

# Start and End index
START_INDEX = 8
NSTEPS = 1

# Core analysis parameters
TH                = 60
LMAX              = 12
N2D               = 3
SPLINE_MINCOUNT   = 7
COUNT_MASK        = 2
MU                = 1e-3
CLIP              = True
NPTS_SUPER        = 101
MCMC              = True
MCMC_WALKERS      = 6
MCMC_STEPS        = 200
MIN_METHOD        = 'L-BFGS-B'
SAVE_FIGS         = True#False
SAVE_PKL          = False