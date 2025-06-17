import numpy as np
from scipy.interpolate import BSpline
from scipy.special import eval_legendre
from scipy.integrate import simps
NAX = np.newaxis

def get_Bsplines_scipy(knots, p, r_grid):
    r"""
    Generates the B-splines :math:`\beta_i(r)` used for 
    discretizing along the radial direction :math:`r = \sqrt{v_{\parallel}^2 + v_{\perp}^2}`
    in FAC velocity phase space.

    Parameters
    ----------
    knots : array-like, shape (n,)
        Array of knot locations in velocity phase space and units of [km/s].

    p : int
        Order of the B-splines.

    r_grid : array-like
        Grid on which the B-spline is evaluated.

    Returns
    -------
    B_i_n : array-like
        The final set of B-splines of shape (knots, radial grid).
    """
    t = np.array([knots[0] for i in range(p)])
    t = np.append(t, knots)
    t = np.append(t, np.array([knots[-1] for i in range(p)]))
    bsp_basis_coefs = np.identity(len(knots) + (p-1))
    spl = BSpline(t, bsp_basis_coefs, p, extrapolate=False)
    B_i_n = np.nan_to_num(spl(r_grid).T)[1:-1]

    return(B_i_n)

def get_Bspline_second_derivative(knots, p, r_grid):
    r"""
    Generates the regularization matrix from the second derivate of B-splines :math:`\partial^2_r \beta_i(r)`.
    This matrix is :math:`\int \partial^2_r \beta_i(r) \, \partial^2_r \beta_j(r) \mathrm{d}r`.

    Parameters
    ----------
    knots : array-like
        Array of knot locations in velocity phase space and units of [km/s].

    p : int
        Order of the B-splines.

    r_grid : array-like
        Grid on which the second derivatives of B-splines are evaluated.

    Returns
    -------
    d2B_i_i : array-like
        The final set of second derivative of B-splines of shape (knots, knots).
    """
    # Evaluate Bsplines on regular grid
    B_i_r = get_Bsplines_scipy(knots, p, r_grid)

    # Define second derivative
    d2r_B_i_r = np.gradient(np.gradient(B_i_r, axis=0), axis=0)
    d2r_B_i_r_sqrd = np.einsum('ij, lj->ilj', d2r_B_i_r, d2r_B_i_r)

    # Integrate
    d2B_i_i = simps(d2r_B_i_r_sqrd * (r**2)[NAX,NAX,:], x=r_grid, axis=-1)

    return(d2B_i_i)

def get_Slepians_scipy(slep_coeffs, theta_grid, Lmax, N2D=None):
    S_alpha_n = None
    theta_nonan = np.radians(theta_grid)

    L = np.arange(0,Lmax+1)
    P_scipy = np.asarray([eval_legendre(ell, np.cos(theta_nonan)) for ell in L])

    # adding the normalization sqrt((2l+1) / 4pi)
    P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]
    S_n_alpha = P_scipy.T @ np.asarray(slep_coeffs)

    # swapping the axes
    S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)
    S_alpha_n = S_alpha_n[:N2D,:]

    return(S_alpha_n)