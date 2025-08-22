import numpy as np
from scipy.optimize import least_squares

def maxwellian_model(vpar, vperp, A, u, wpar, wperp):
    Q = (((vpar - u)**2) / (wpar**2)) + ((vperp**2) / (wperp**2))
    return A * np.exp(-Q)

def _moment_init(vpar, vperp, y):
    """
    Moment-based initializer using y as weights.
    w relates to std dev by: w = sqrt(2) * sigma.
    """
    vpar = vpar.ravel(); vperp = vperp.ravel(); y = y.ravel()
    w = np.clip(y, 0, None)
    s = w.sum()
    if s <= 0:
        # fallback: unweighted moments
        w = np.ones_like(y); s = w.size
    u0 = (w * vpar).sum() / s
    var_par = (w * (vpar - u0)**2).sum() / s
    var_perp = (w * (vperp)**2).sum() / s
    # guard against degenerate widths
    wpar0 = max(np.sqrt(2.0 * max(var_par, 1e-12)), 1e-6)
    wperp0 = max(np.sqrt(2.0 * max(var_perp, 1e-12)), 1e-6)
    A0 = float(np.max(y))
    if not np.isfinite(A0) or A0 <= 0:
        A0 = 1.0
    return A0, u0, wpar0, wperp0

def fit_maxwellian(vpar, vperp, ubulk, y, weights=None, init=None, robust=False, f_scale=1.0):
    """
    Fit A * exp(-((vpar - u)^2)/wpar^2 - (vperp^2)/wperp^2) to data y.
    
    Parameters
    ----------
    vpar, vperp, y : array-like (same shape)
        Sample locations and data values.
    weights : array-like, optional
        Per-point weights (larger => more influence). Applied to residuals.
    init : tuple (A0, u0, wpar0, wperp0), optional
        Initial guess. If None, uses moment-based initializer.
    robust : bool, default False
        If True, uses soft-L1 loss (robust). Else ordinary least squares.
    f_scale : float, default 1.0
        Scaling for robust loss.

    Returns
    -------
    result : dict
        Keys: A, u, wpar, wperp, success, message, cost, cov (approx), nfev
    """
    vpar = np.asarray(vpar).ravel()
    vperp = np.asarray(vperp).ravel()
    y = np.asarray(y).ravel()
    n = y.size

    if init is None:
        A0, u0, wpar0, wperp0 = _moment_init(vpar, vperp, y)
    else:
        A0, u0, wpar0, wperp0 = init

    def model(params):
        A, wpar, wperp = params
        return maxwellian_model(vpar, vperp, A, ubulk, wpar, wperp)

    def residuals(params):
        r = model(params) - y
        if weights is not None:
            r = np.sqrt(np.asarray(weights).ravel()) * r
        return r

    def jacobian(params):
        A, u, wpar, wperp = params
        Q = ((vpar - u)**2) / (wpar**2) + (vperp**2) / (wperp**2)
        E = np.exp(-Q)               # so f = A*E
        f = A * E
        # partials for residuals = f - y
        d_dA    = E
        d_du    = f * (2.0 * (vpar - u) / (wpar**2))
        d_dwpar = f * (2.0 * (vpar - u)**2 / (wpar**3))
        d_dwperp= f * (2.0 * (vperp**2)     / (wperp**3))
        J = np.column_stack([d_dA, d_du, d_dwpar, d_dwperp])
        if weights is not None:
            W = np.sqrt(np.asarray(weights).ravel())[:, None]
            J = W * J
        return J

    # Bounds enforce positivity of A, wpar, wperp
    eps = 1e-10
    lb = np.array([eps, -np.inf, eps, eps])
    ub = np.array([np.inf, np.inf, np.inf, np.inf])

    x0 = np.array([max(A0, eps), u0, max(wpar0, eps), max(wperp0, eps)])

    res = least_squares(
        residuals, x0=x0, jac=jacobian, bounds=(lb, ub),
        loss=('soft_l1' if robust else 'linear'), f_scale=f_scale,
        x_scale='jac', max_nfev=2000
    )

    # Approximate covariance (LS theory)
    dof = max(1, n - 4)
    cov = None
    try:
        JTJ = res.jac.T @ res.jac
        sigma2 = (res.cost * 2.0) / dof  # res.cost = 0.5*||r||^2
        cov = np.linalg.inv(JTJ) * sigma2
    except Exception:
        pass

    A, u, wpar, wperp = res.x
    return dict(A=float(A), u=float(u), wpar=float(wpar), wperp=float(wperp),
                success=bool(res.success), message=res.message, cost=float(res.cost),
                nfev=res.nfev, cov=cov)

def get_M_Mrec(gvdf_tstamp, y, density_percent=1.0):
    # concatenating to make both sides
    vpara = np.concatenate([gvdf_tstamp.vpara_nonan_inst, gvdf_tstamp.vpara_nonan_inst])
    vperp = np.concatenate([-gvdf_tstamp.vperp_nonan_inst, gvdf_tstamp.vperp_nonan_inst])
    y = np.concatenate([y, y])

    M = fit_maxwellian(vpara, vperp, y)
    M_rec = maxwellian_model(vpara, vperp, M['A'], M['u'], M['wpar'], M['wperp'])

    return M, M_rec

def convert_f_to_logscaledf(fdata, gvdf_tstamp):
    # concatenating to make both sides
    vpara = np.concatenate([gvdf_tstamp.vpara_nonan_inst, gvdf_tstamp.vpara_nonan_inst])
    vperp = np.concatenate([-gvdf_tstamp.vperp_nonan_inst, gvdf_tstamp.vperp_nonan_inst])
    y = np.concatenate([fdata, fdata])
    # fitting the maxwellian
    gvdf_tstamp.M = fit_maxwellian(vpara, vperp, np.linalg.norm(gvdf_tstamp.ubulk), y)   
    # reconstructing
    gvdf_tstamp.M_rec = maxwellian_model(gvdf_tstamp.vpara_nonan_inst, gvdf_tstamp.vperp_nonan_inst,
                                         gvdf_tstamp.M['A'], gvdf_tstamp.M['u'], gvdf_tstamp.M['wpar'], gvdf_tstamp.M['wperp'])

    # finding the maximum difference of data - M_rec
    gvdf_tstamp.epsilon = np.nanmin(fdata - gvdf_tstamp.M_rec)

    # log of the scaled data
    log_scaled_f = np.log10((fdata + 1) / (gvdf_tstamp.M_rec - gvdf_tstamp.epsilon + 1))

    gvdf_tstamp.log_minval = np.nanmin(log_scaled_f)

    return log_scaled_f - gvdf_tstamp.log_minval

def convert_logresf_to_f(log_scaled_f_norm, gvdf_tstamp, vpara, vperp):
    M = maxwellian_model(vpara, vperp,
                         gvdf_tstamp.M['A'], gvdf_tstamp.M['u'], gvdf_tstamp.M['wpar'], gvdf_tstamp.M['wperp'])

    log_scaled_f = log_scaled_f_norm + gvdf_tstamp.log_minval
    scaled_f = np.power(10, log_scaled_f)
    f = scaled_f * (M - gvdf_tstamp.epsilon + 1) - 1

    return f
