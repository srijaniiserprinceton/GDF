# --- same imports & data generation as you already have ---

from numpy.polynomial.legendre import legval, legder

# Monotone-decreasing Legendre fit (pure NumPy, robust active set + derivative constraints)
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from numpy.polynomial.legendre import legval, legder

# ---------- 1) Generate data ----------
rng = np.random.default_rng(42)
n = 80
x = np.linspace(-1, 1, n)

# Non-monotone data (rises then falls), but we will enforce a monotone-decreasing fit
f_true = 1.7 - 0.9*x - 0.18*np.exp(3.2*x) - 3.0*x**2
wiggle  = -0.18 * np.exp(-((x + 0.92)/0.06)**2)  # local dip near left edge
noise   = 0.05 * rng.standard_normal(n)
y = f_true + wiggle + noise

def Ldesign(xs, deg):
    xs = np.asarray(xs)
    Phi = np.zeros((xs.size, deg+1))
    for k in range(deg+1):
        c = np.zeros(deg+1); c[k] = 1.0
        Phi[:, k] = legval(xs, c)
    return Phi

def Ldesign_deriv(xs, deg, order=1):
    xs = np.asarray(xs)
    D = np.zeros((xs.size, deg+1))
    for k in range(deg+1):
        c = np.zeros(deg+1); c[k] = 1.0
        dc = legder(c, m=order)
        D[:, k] = legval(xs, dc) if dc.size else 0.0
    return D

deg = 6
Phi = Ldesign(x, deg)

# ---- unconstrained LS (unchanged) ----
coef_ls, *_ = np.linalg.lstsq(Phi, y, rcond=None)

# ---- derivative constraints with margin tau ----
xg  = np.linspace(-1, 1, 2000)             # denser constraint grid
Bp  = Ldesign_deriv(xg, deg)
A   = -Bp                                   # -f'(xg) >= b
tau = 1e-6                                  # margin to absorb roundoff
b    = -tau * np.ones(A.shape[0])           # allow up to +tau increase

# ---- robust active-set with line-search (same logic, minor tuning) ----
def constrained_ls_monotone_deriv(Phi, y, A, b, ridge=1e-7, eps=1e-9, max_iter=3000):
    col_scale = np.linalg.norm(Phi, axis=0); col_scale[col_scale==0]=1.0
    Phis = Phi/col_scale; As = A/col_scale
    H = Phis.T @ Phis + ridge*np.eye(Phis.shape[1])
    g = Phis.T @ y
    m = np.linalg.solve(H, g)               # start LS
    active = np.array([], dtype=int)

    for _ in range(max_iter):
        s = As @ m - b                      # slack
        viol = np.where(s < -eps)[0]
        if viol.size:
            add = viol[np.argmin(s[viol])]
            active = np.unique(np.r_[active, add])

        if active.size:
            At = As[active,:]
            KKT = np.block([[H, At.T], [At, np.zeros((At.shape[0],At.shape[0]))]])
            rhs = np.concatenate([g, b[active]])
            sol = np.linalg.solve(KKT, rhs)
            m_eq = sol[:Phis.shape[1]]
            lam  = sol[Phis.shape[1]:]

            p = m_eq - m
            if np.linalg.norm(p) > 0:
                Ap = As @ p
                mask = Ap < 0
                alpha = 1.0
                if np.any(mask):
                    alpha = float(min(1.0, 0.999999*np.min(s[mask]/(-Ap[mask]))))
                m = m + alpha*p
                if alpha < 1.0:
                    s = As @ m - b
                    hit = np.where(np.isclose(s, 0.0, atol=5e-12))[0]
                    active = np.unique(np.r_[active, hit])
                    continue

            neg = np.where(lam < -eps)[0]
            if neg.size:
                drop = active[neg[np.argmin(lam[neg])]]
                active = active[active != drop]
                continue

        if not viol.size:
            break

    return m/col_scale, active

coef_mon, _ = constrained_ls_monotone_deriv(Phi, y, A, b)

# ---- evaluate & *robust* monotonicity check ----
xx    = np.linspace(-1, 1, 2000)
Bxx   = Ldesign(xx, deg)
Bp_xx = Ldesign_deriv(xx, deg)

y_ls  = Bxx @ coef_ls
y_mon = Bxx @ coef_mon

max_pos_slope = np.max(Bp_xx @ coef_mon)     # should be <= tau-ish
mono_ok = max_pos_slope <= tau + 5e-7        # same order as margin

print("max positive slope =", max_pos_slope)

# ---- plots (same as before) ----
