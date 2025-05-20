# init_parameters.py

# Define the time range we are running for
TRANGE = ['2020-01-26T07:12:00', '2020-01-26T07:30:59']

# Get the credentials
creds = None

# Core analysis parameters
TH                = 60
LMAX              = 12
N2D               = 3
P                 = 3
SPLINE_MINCOUNT   = 7
COUNT_MASK        = 2
ITERATE           = False
CLIP              = True
MCMC              = True
MCMC_WALKERS      = 4
MCMC_STEPS        = 800
MIN_METHOD        = 'L-BFGS-B'
SAVE_FIGS         = False