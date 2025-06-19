.. _getting_started:

***************
Getting started
***************

The ``gdf`` repository directory comes with an ``init_gdf_default.py`` initialization file. It is 
expected that the user would only need to change (normally, just a few) of the parameters in this initialization
file to obtain the distributions and associated moments in a desired time interval.

.. literalinclude:: ../../init_gdf_default.py
   :language: python
   :lines: 1-28

.. list-table:: Description of parameters in the ``init_gdf_default.py`` initialization file.
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - TRANGE
     - list of str
     - Start and end times in ISO 8601 standard format in UTC ``YYYY-MM-DD-Thh:mm:ss``
   * - CREDS
     - str or ``None``
     - Path to a ``json`` file containing credentials to access SWEAP and FIELDS propreitary data.
   * - START_INDEX
     - int
     - Index of first desired time stamp for ``gdf`` with respect to the start time in TRANGE.
   * - NSTEPS
     - int or ``None``
     - Number of timestamps to analyze starting at START_INDEX. Use ``None`` for entire TRANGE.
   * - TH
     - float
     - Polar cap extent in degrees about the magnetic field axis used for generating Slepians.
   * - LMAX
     - int
     - Maximum angular degree for constructing the 1D Slepian basis on a polar cap..
   * - N2D
     - int or ``None``
     - The number of polar-cap Slepian functions to be used for reconstructions.
   * - P
     - int (default: 3)
     - The degree of the B-spline used for discretizing in velocity phase space.
   * - SPLINE_MINCOUNT
     - int
     - Minimum number of grid points to be contained in each B-spline shell.
   * - COUNT_MASK
     - int
     - Particle count threshold below which grid points are ignored from reconstructions.
   * - MU
     - float
     - Regularization parameter for gyroaxis optimization, super-resolution uses L-curve.
   * - CLIP
     - bool
     - <What on earth was this parameter?>
   * - NPTS_SUPER
     - int
     - Number of super-resolved points in a uniform Cartesian grid of (:math:`v_{\parallel},\, v_{\perp}`).
   * - MCMC
     - bool
     - Flag to use a MCMC refinement of the gyroaxis, provides error esimtates on the gyro-centroid.
   * - MCMC_WALKERS
     - int
     - Number of MCMC walkers to be used. Need to use 3 minimum walkers.
   * - MCMC_STEPS
     - int
     - Number of MCMC steps for each walker. Optimal choice: (6 MCMC_WALKERS, 200 MCMC_STEPS).
   * - MIN_METHOD
     - str
     - Minimization method used in the ``scipy.optimize.minimize``. Optimal choice is 'L-BFGS-B'.
   * - SAVE_FIGS
     - bool
     - Flag if the final (and diagnostic) figures should be saved under directories in ``Figures/``.
   * - SAVE_PKL
     - bool
     - Flag if the final metadata should be stored as a ``pkl`` file in the ``Outputs/`` directory.