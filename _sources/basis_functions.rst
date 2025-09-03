.. _basis_functions:

******************************
Description of basis functions
******************************

Our approach of reconstructing the GDF from SPAN-Ai measurements involves two steps:

* Estimating the gyro-centroid :math:`\boldsymbol{v}_{\mathrm{gyro}}` through which the magnetic field :math:`\boldsymbol{B}` is anchored.

* Inferring the gyrotropic distribution which best matches the FAC grids projected in the :math:`(v_{\parallel},\, v_{\perp})` plane. 

The first step uses a Bayesian framework, which requires using a faster Slepian functions generation
algorithm. This is why we use a 1D Slepian on-a-polar-cap (in angular space :math:`\theta`) along with 1D cubic B-splines
in radial space :math:`r = \sqrt{v_{\parallel}^2 + v_{\perp}^2}`. The overall setup for the polar cap
fitting method is shown in :numref:`fitting-setup`.

.. figure:: /_static/images/Fitting_setup.png
   :name: fitting-setup
   :alt: Fitting setup for polar cap method.
   :align: center
   :width: 95%

   Fitting architecture when using 1D Slepians on a polar cap. 
   Panel (A) shows the distribution of grid points in black after 
   rotating to FAC followed by boosting the frame to induce a maximum 
   angle of :math:`\Theta` to the grid points about the origin.
   The set of B-splines corresponding to the SPAN-Ai grids projected into FAC 
   is shown in panel (B). The local support in radius for the red
   B-spline highlighted is shown as the shaded red arc in panel (A). 
   All grid points which encompassed by this B-spline is marked with a 
   larger size. Panel (C) shows the Slepian functions optimally concentrated 
   within angle :math:`\Theta` about the gyroaxis. The distribution of the grid 
   points within the red arc of panel (A) is overplotted on each of the 
   Slepian functions :math:`S_{\alpha}(\theta)`. The vertical red dashed
   line marks :math:`\Theta` used for the polar cap shown in panel (A).

Radial knots for cubic B-splines
================================

The domain is discretized as a function of velocity :math:`v` and polar angle :math:`\theta`. 
We use cubic B-splines with local supports at knots, which are computed for each timestamp, 
to render localized support on different velocity shells. The knots are placed 
logarithmically with a spacing of :math:`\Delta \log_{10}(v)=`. 
We arrive at this number after investigating the average log spacing 
between SPAN-i velocity shells as 

.. math::
    \Delta \log_{10}(v) = \frac{1}{N_{\rm{shells}}}\sum_{i = 0}^{N_{\rm{shells}}} \log_{10}(v_{i+1}) - \log_{10}(v_i) \, .

The number of knots are calculated based on this logarithmic spacing and the farthest extents of the grids which have a non-zero count.

.. math::
    N_{\rm{knots}} \simeq \frac{\log_{10}(v_{\rm{max}}) - \log_{10}(v_{\rm{min}})}{\Delta \log_{10}(v)} \, ,

where, :math:`N_{\rm{knots}}` is rounded down to the closest integer and 
:math:`(v_{\rm{min}}, v_{\rm{max}})` are the minimum and maximum velocity 
magnitudes of grids with non-zero counts. Finally, the knot locations 
in :math:`v` are computed from binning :math:`\log_{10}(v)` into :math:`N_{\rm{knots}}` 
and finding the bin centers. The above prescription renders the first 
and the last points at the edge of the domain, resulting in zero B-spline support. 
In order to ensure that the very first and last points are supported by B-splines, 
we add two additional knots at :math:`\log_{10}(v_{\rm{min}}) - \Delta \log_{10}(v)` 
and :math:`\log_{10}(v_{\rm{max}}) + \Delta \log_{10}(v)`. Finally, these knots are 
raised to the power of 10 and are used to generate the B-splines. 
These knots are computed for each timestamp as the grids change. 
The resultant B-splines for our chosen timestamp is shown in panel 
(B) of :numref:`fitting-setup`. We have highlighted one of the 
B-spline peaking around :math:`v \sim 470` km/s in red. 
The arc of :math:`(v_{\parallel}, v_{\perp})` space is also shaded in 
red in panel (A). The cloud of grids which are spanned by this 
B-spline (using 0.1 in panel (B) as a threshold) are marked in 
large black dots in panel (A).

.. autofunction:: basis_funcs.get_Bsplines_scipy

Polar-cap Slepians from localization coefficients
=================================================

In panel (C) of :numref:`fitting-setup`, we show the three Slepian 
functions which are optimally confined inside a :math:`\Theta = 60^{\circ}` 
polar cap with a maximum wavenumber :math:`L_{\rm{max}} = 12`. These functions 
capture the variation of the distribution across the grid points as a function 
of polar angle for each B-spline velocity shell. The :math:`\Theta = 60^{\circ}` 
is demarcated by a vertical red dashed line. The angular location of the 
grids spanned within the red arc in panel (A) are over-plotted on each Slepian function.

.. autofunction:: basis_funcs.get_Slepians_scipy