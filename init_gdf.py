# init_gdf.py

# Define the time range we are running for
# TRANGE = ['2020-01-26T07:12:00', '2020-01-26T07:30:59']
TRANGE = ['2022-02-25T16:56:00', '2022-02-25T17:01:00']
# TRANGE = ['2024-03-30T12:12:00', '2024-03-30T17:30:59']
# TRANGE = ['2024-09-24T12:12:00', '2024-09-24T17:30:59']

# Get the credentials
CREDS = None       # credential file.

# Start and End index
START_INDEX = 0
NSTEPS = None

# Core analysis parameters
TH                = 60
LMAX              = 12
N2D               = 3
P                 = 3
SPLINE_MINCOUNT   = 7
COUNT_MASK        = 2
MU                = 1e-3   # only used for gyroaxis and not superresolution
CLIP              = True
NPTS_SUPER        = 101
MCMC              = False #True
MCMC_WALKERS      = 8
MCMC_STEPS        = 2000
MIN_METHOD        = 'L-BFGS-B'
SAVE_FIGS         = True
SAVE_PKL          = True