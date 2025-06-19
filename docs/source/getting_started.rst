.. _getting_started:

***************
Getting started
***************

The `gdf` repository directory comes

.. code-block:: python

    # init_gdf.py

    # Define the time range we are running for
    TRANGE = ['2020-01-26T14:28:00', '2020-01-26T15:30:59']

    # file containing credentials, if accessing proprietary data
    CREDS = None

    # Start and End index
    START_INDEX = 31
    NSTEPS = 1          # use None when running for entire TRANGE

    # Core analysis parameters
    TH                = 30
    LMAX              = 12
    N2D               = 3
    P                 = 3
    SPLINE_MINCOUNT   = 7
    COUNT_MASK        = 2
    MU                = 1e-3   # only used for gyroaxis and not superresolution
    CLIP              = True
    NPTS_SUPER        = 101
    MCMC              = True
    MCMC_WALKERS      = 6
    MCMC_STEPS        = 200
    MIN_METHOD        = 'L-BFGS-B'
    SAVE_FIGS         = True
    SAVE_PKL          = True