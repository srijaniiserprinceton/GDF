import sys
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import eval_legendre
import matplotlib.pyplot as plt; plt.ion()

import eval_Slepians

NAX = np.newaxis

# generating a B-spline basis
Slep = eval_Slepians.Slep_transverse()
Slep.gen_Slep_tapers(75, 12)
S_alpha_n = None

theta_dense = np.linspace(0, 180, 1000)
Lmax = 12

def gen_Slep_basis(theta_grid):
    L = np.arange(0,Lmax+1)
    P_scipy = np.asarray([eval_legendre(ell, np.cos(theta_grid * np.pi / 180)) for ell in L])
    # adding the normalization sqrt((2l+1) / 4pi)
    P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]
    S_n_alpha = P_scipy.T @ np.asarray(Slep.C)

    # swapping the axes
    S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)
    # truncating beyond Shannon number
    N2D = int(np.sum(Slep.V)) - 1
    S_alpha_n = S_alpha_n[:N2D,:]

    return S_alpha_n, N2D

S_alpha_n_dense, N2D = gen_Slep_basis(theta_dense)

plt.figure()
for i in range(N2D): plt.plot(theta_dense, S_alpha_n_dense[i])
plt.axvline(75)


# plotting the data after excluding some points from the center
xgrid = np.linspace(0, 180, 1000)
sigma = 20
data = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * (xgrid / sigma)**2)
plt.figure()
plt.plot(xgrid, data)
plt.axvline(75)

# reference coefficients
# coeffs_ref = np.array([ 1.21352956e-02, -1.37743799e-03,  3.27022743e-04, -6.52798704e-05])
coeffs_ref = np.array([ 1e0, 1e-1,  1e-2, 1e-3])
# coeffs_ref = np.ones_like(coeffs_ref)

# generating on a sparse grid
theta_sparse = np.linspace(0, 75, 10)[6:]
data_sparse = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * (theta_sparse / sigma)**2)

# generating Slepians on the same grid
S_alpha_n_sparse, N2D = gen_Slep_basis(theta_sparse)

S_alpha_n_sparse = coeffs_ref[:,NAX] * S_alpha_n_sparse

data_misfit = []
model_misfit = []

lambda_arr = np.linspace(-7,-2,100)

for lam in lambda_arr:
    # reconstructing using Slepian functions
    GTG = S_alpha_n_sparse @ S_alpha_n_sparse.T + np.power(10., lam) * np.identity(S_alpha_n_sparse.shape[0])

    coeffs = np.linalg.inv(GTG) @ S_alpha_n_sparse @ data_sparse

    # reconstructing
    data_rec_sparse = coeffs @ S_alpha_n_sparse

    # now super-resolving
    data_rec_dense = coeffs @ (coeffs_ref[:,NAX] * S_alpha_n_dense)

    # plt.figure()
    # plt.plot(theta_dense, data, 'r')
    # plt.plot(theta_sparse, data_rec_sparse, 'ok')
    # plt.plot(theta_sparse, data_sparse, 'xr')
    # plt.plot(theta_dense, data_rec_dense, '--k')
    # plt.axvline(75)

    data_misfit.append(np.linalg.norm((data_sparse - data_rec_sparse)))
    model_misfit.append(np.linalg.norm(coeffs))

data_misfit = np.asarray(data_misfit)
model_misfit = np.asarray(model_misfit)

plt.figure()
plt.plot(model_misfit, data_misfit, '.k')

def compute_lcurve_corner(norms, residuals):
    """
    Find the 'corner' of the L-curve (max curvature) in log-log space.
    Returns the index of the optimal lambda.
    """
    # Convert to log-log space for better curvature detection
    x = np.log10(norms)
    y = np.log10(residuals)
    # x = norms
    # y = residuals

    # Compute first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Compute curvature κ using parametric form
    curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5

    # Find the index of max curvature
    knee_index = np.argmax(curvature)
    return knee_index

def geometric_knee(x, y):
    x, y = np.array(x), np.array(y)
    # Line: from first to last point
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    point_vecs = np.stack([x - x[0], y - y[0]], axis=1)
    proj_lens = point_vecs @ line_vec_norm
    proj_points = np.outer(proj_lens, line_vec_norm) + np.array([x[0], y[0]])
    distances = np.linalg.norm(point_vecs - (proj_points - np.array([x[0], y[0]])), axis=1)
    return np.argmax(distances)


knee_idx = (len(lambda_arr) - 1) - geometric_knee(data_misfit, model_misfit)

# knee_idx = len(lambda_arr) - compute_lcurve_corner(norms=model_misfit, residuals=data_misfit)

plt.plot(model_misfit[knee_idx], data_misfit[knee_idx], 'xr')

GTG = S_alpha_n_sparse @ S_alpha_n_sparse.T + 1.0 * np.power(10., lambda_arr[knee_idx]) * np.identity(S_alpha_n_sparse.shape[0])

coeffs = np.linalg.inv(GTG) @ S_alpha_n_sparse @ data_sparse

# reconstructing
data_rec_sparse = coeffs @ S_alpha_n_sparse

# now super-resolving
data_rec_dense = coeffs @ (coeffs_ref[:,NAX] * S_alpha_n_dense)

plt.figure()
plt.plot(theta_dense, data, 'r')
plt.plot(theta_sparse, data_rec_sparse, 'ok')
plt.plot(theta_sparse, data_sparse, 'xr')
plt.plot(theta_dense, data_rec_dense, '--k')
plt.axvline(75)

import numpy as np
from sklearn.linear_model import LinearRegression

def fit_line(x, y):
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def intersection_point(m1, b1, m2, b2):
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

def find_lcurve_corner(x_vals, y_vals, split_index):
    # x = np.log10(x_vals)
    # y = np.log10(y_vals)
    x = x_vals
    y = y_vals

    # Split L-curve into left and right segments
    x1, y1 = x[:split_index], y[:split_index]
    x2, y2 = x[-split_index:], y[-split_index:]

    m1, b1 = fit_line(x1, y1)
    m2, b2 = fit_line(x2, y2)

    # plotting the line
    xline = np.linspace(x.min(), x.max(), 100)
    yline1 = m1 * xline + b1 
    yline2 = m2 * xline + b2

    # Compute intersection point of the two lines
    x_int, y_int = intersection_point(m1, b1, m2, b2)

    # Compute distance of all points to the intersection
    distances = np.sqrt((x - x_int)**2 + (y - y_int)**2)
    min_idx = np.argmin(distances)

    return min_idx, xline, yline1, yline2, x_int, y_int

model_misfit = (model_misfit - model_misfit.min()) / (model_misfit.max() - model_misfit.min())
data_misfit = (data_misfit - data_misfit.min()) / (data_misfit.max() - data_misfit.min())

knee_idx, xline, yline1, yline2, x_int, y_int = find_lcurve_corner(model_misfit, data_misfit, 10)
plt.figure()
plt.plot(model_misfit, data_misfit, '.k')
plt.plot(model_misfit[knee_idx], data_misfit[knee_idx], 'xr')
plt.plot(xline, yline1, 'k', alpha=0.3)
plt.plot(xline, yline2, 'k', alpha=0.3)
plt.plot(x_int, y_int, 'ok')

knee_idx = len(lambda_arr) - knee_idx

GTG = S_alpha_n_sparse @ S_alpha_n_sparse.T + np.power(10., lambda_arr[knee_idx]) * np.identity(S_alpha_n_sparse.shape[0])

coeffs = np.linalg.inv(GTG) @ S_alpha_n_sparse @ data_sparse

# reconstructing
data_rec_sparse = coeffs @ S_alpha_n_sparse

# now super-resolving
data_rec_dense = coeffs @ (coeffs_ref[:,NAX] * S_alpha_n_dense)

plt.figure()
plt.plot(theta_dense, data, 'r')
plt.plot(theta_sparse, data_rec_sparse, 'ok')
plt.plot(theta_sparse, data_sparse, 'xr')
plt.plot(theta_dense, data_rec_dense, '--k')
plt.axvline(75)


sys.exit()
# performing mirroring 
# generating Slepians on the same grid
S_alpha_n_sparse, N2D = gen_Slep_basis(theta_sparse)

# mirroring
theta_sparse = np.hstack([-np.flip(theta_sparse), theta_sparse])
S_alpha_n_sparse = np.hstack([np.flip(S_alpha_n_sparse, axis=1), S_alpha_n_sparse])
data_sparse = np.hstack([np.flip(data_sparse), data_sparse])

# reconstructing using Slepian functions
GTG = S_alpha_n_sparse @ S_alpha_n_sparse.T

coeffs = np.linalg.inv(GTG) @ S_alpha_n_sparse @ data_sparse

# reconstructing
data_rec_sparse = coeffs @ S_alpha_n_sparse

# now super-resolving
data = np.hstack([np.flip(data), data])
theta_dense = np.hstack([-np.flip(theta_dense), theta_dense])
S_alpha_n_dense = np.hstack([np.flip(S_alpha_n_dense, axis=1), S_alpha_n_dense])
data_rec_dense = coeffs @ S_alpha_n_dense

plt.figure()
plt.plot(theta_dense, data, 'r')
plt.plot(theta_sparse, data_rec_sparse, 'ok')
plt.plot(theta_sparse, data_sparse, 'xr')
plt.plot(theta_dense, data_rec_dense, '--k')
plt.axvline(75)
plt.axvline(-75)
