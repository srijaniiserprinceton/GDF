import numpy as np
from scipy.interpolate import BSpline
from scipy.special import eval_legendre

NAX = np.newaxis

def get_Bsplines_scipy(knots, p, r_grid):
    t = np.array([knots[0] for i in range(p)])
    t = np.append(t, knots)
    t = np.append(t, np.array([knots[-1] for i in range(p)]))
    bsp_basis_coefs = np.identity(len(knots) + (p-1))
    spl = BSpline(t, bsp_basis_coefs, p, extrapolate=False)
    B_i_n = spl(r_grid).T
    B_i_n = np.nan_to_num(spl(r_grid).T)

    return(B_i_n)

def get_Bspline_second_derivative(B_i_n):
    B_i_n_rr = np.gradient(np.gradient(B_i_n, axis=0), axis=0)
    B_i_n_rr_sqrd = np.einsum('ij, lj->ilj', B_i_n_rr, B_i_n_rr)
    return(B_i_n_rr_sqrd)

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