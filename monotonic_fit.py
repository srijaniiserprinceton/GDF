# Monotone-decreasing Legendre fit demo (with an initial non-monotone wiggle in the data)
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from numpy.polynomial.legendre import legval

# ---------- 1) Generate data ----------
rng = np.random.default_rng(42)
n = 80
x = np.linspace(-1, 1, n)

# strictly decreasing "true" curve on [-1, 1]
f_true = 1.7 - 0.9*x - 0.18*np.exp(3.2*x) - 3 * x**2

# add a small local dip near the far left so the earliest points are non-monotone
wiggle = -0.18 * np.exp(-((x + 0.92)/0.06)**2)  # localized negative bump at x≈-0.92
noise = 0.05 * rng.standard_normal(n)
y = f_true + wiggle + noise

# ---------- 2) Legendre design matrix ----------
def legendre_design(xs, deg):
    xs = np.asarray(xs)
    Phi = np.zeros((xs.size, deg+1))
    for k in range(deg+1):
        c = np.zeros(deg+1); c[k] = 1.0
        Phi[:, k] = legval(xs, c)
    return Phi

deg = 6
Phi = legendre_design(x, deg)

# ---------- 3) Unconstrained least squares ----------
coef_ls, *_ = np.linalg.lstsq(Phi, y, rcond=None)

# ---------- 4) Build monotone-decreasing constraints ----------
# Use first-difference constraints on a dense grid: (B @ m)[j] - (B @ m)[j+1] >= 0
xg = np.linspace(-1, 1, 300)
Bg = legendre_design(xg, deg)

def first_diff_matrix(m):
    F = np.zeros((m-1, m))
    for j in range(m-1):
        F[j, j]   =  1.0
        F[j, j+1] = -1.0
    return F

F = first_diff_matrix(len(xg))
A = F @ Bg
b = np.zeros(A.shape[0])

# ---------- 5) Constrained LS via a tiny active-set (KKT) solver ----------
def constrained_ls_monotone(Phi, y, A, b, ridge=1e-12, eps=1e-10, max_iter=200):
    H = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    g = Phi.T @ y
    # start from unconstrained solution
    m = np.linalg.solve(H, g)
    active = np.array([], dtype=int)
    for _ in range(max_iter):
        # Check primal feasibility
        s = A @ m - b
        viol = np.where(s < -eps)[0]
        if viol.size:
            # add the most violated constraint
            add = viol[np.argmin(s[viol])]
            active = np.unique(np.r_[active, add])
        # Solve KKT for active set
        if active.size:
            At = A[active, :]
            KKT = np.block([[H, At.T],
                            [At, np.zeros((At.shape[0], At.shape[0]))]])
            rhs = np.concatenate([g, b[active]])
            sol = np.linalg.solve(KKT, rhs)
            m = sol[:Phi.shape[1]]
            lam = sol[Phi.shape[1]:]
            # Drop most negative multiplier (dual infeasibility)
            neg = np.where(lam < -eps)[0]
            if neg.size:
                drop_idx = active[neg[np.argmin(lam[neg])]]
                active = active[active != drop_idx]
                continue
        # If no violations and dual feasible (or no active), we're done
        if not viol.size:
            break
    return m, active

coef_mon, active_rows = constrained_ls_monotone(Phi, y, A, b)

# ---------- 6) Evaluate and check monotonicity ----------
xx = np.linspace(-1, 1, 400)
Bplot = legendre_design(xx, deg)
y_ls  = Bplot @ coef_ls
y_mon = Bplot @ coef_mon

def is_monotone_decreasing(vals, tol=1e-10):
    return np.all(np.diff(vals) <= tol)

mono_ok = is_monotone_decreasing(legendre_design(xg, deg) @ coef_mon)

# ---------- 7) Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# (a) Unconstrained fit
axes[0].scatter(x, y, s=16, alpha=0.8, label="data")
axes[0].plot(xx, y_ls, linewidth=2, label="unconstrained fit")
axes[0].set_title("Unconstrained Legendre LS")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].legend()

# (b) Monotone-decreasing constrained fit
axes[1].scatter(x, y, s=16, alpha=0.8, label="data")
axes[1].plot(xx, y_mon, linewidth=2, label="monotone-decreasing fit")
axes[1].set_title(f"Constrained (mono↓) • check={mono_ok}")
axes[1].set_xlabel("x"); axes[1].legend()

fig.tight_layout()