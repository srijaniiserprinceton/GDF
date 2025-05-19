import sys
from tqdm import tqdm
import numpy as np
from astropy.coordinates import cartesian_to_spherical as c2s
import matplotlib.pyplot as plt; plt.ion()
from line_profiler import profile
from scipy.interpolate import BSpline
from scipy.special import eval_legendre
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
NAX = np.newaxis

import eval_Slepians
import src.functions as fn

mu = 1e-3

def get_rho_z(points, p0, n):
    # Align coordinates with axis
    x_rel = points - p0
    z = np.dot(x_rel, n)
    x_proj = x_rel - np.outer(z, n)
    rho = np.linalg.norm(x_proj, axis=1)
    return rho, z

def fit_axisymmetric_model(rho, z, values, degree=3):
    X = np.vstack([rho, z]).T
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-1))
    model.fit(X, values)
    pred = model.predict(X)
    return pred, model

def loss_fn_Polynomials(p0_2d, points, values, n, origin, u, v):
    p0 = origin + p0_2d[0]*u + p0_2d[1]*v
    rho, z = get_rho_z(points, p0, n)
    pred, _ = fit_axisymmetric_model(rho, z, values)
    return np.mean((values - pred)**2)

def loss_fn_Slepians(p0_2d, points, values, n, origin, u, v):
    p0 = origin + p0_2d[0]*u + p0_2d[1]*v
    pred = gvdf_tstamp.inversion(p0, values, tidx)
    return np.mean((values - pred)**2)

@profile
def find_symmetry_point(points, values, n, loss_fn, origin=None):
    # Get basis u, v orthogonal to n
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(arbitrary, n):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    if(origin is None): origin = np.mean(points, axis=0)  # reasonable guess
    res = minimize(loss_fn, x0=[0.0, 0.0], args=(points, values, n, origin, u, v), method='Powell')
    best_p0 = origin + res.x[0] * u + res.x[1] * v
    return best_p0, res.fun

def point_on_axis_with_x(p0, n, target_x):
    if np.isclose(n[0], 0):
        raise ValueError("Axis is perpendicular to x-direction; cannot target a specific x.")
    t = (target_x - p0[0]) / n[0]
    return p0 + t * n

def merge_bins(bin_edges, counts, threshold):
    merged_edges = []
    merged_counts = []

    current_count = 0
    start_edge = bin_edges[0]

    for i in range(len(counts)):
        current_count += counts[i]

        # If merged count is at or above threshold, finalize the current bin
        if current_count >= threshold:
            end_edge = bin_edges[i + 1]
            merged_edges.append((start_edge, end_edge))
            merged_counts.append(current_count)
            if i + 1 < len(bin_edges):  # Prepare for next merge
                start_edge = bin_edges[i + 1]
            current_count = 0
        # else continue merging into the next bin

    # Handle any remaining counts (less than threshold at end)
    if current_count > 0:
        if merged_edges:
            # Merge remaining with last bin
            last_start, last_end = merged_edges[-1]
            merged_edges[-1] = (last_start, bin_edges[-1])
            merged_counts[-1] += current_count
        else:
            # If everything was under threshold, merge all into one
            merged_edges.append((bin_edges[0], bin_edges[-1]))
            merged_counts.append(current_count)

    return merged_edges, merged_counts

class gyrovdf:
    def __init__(self, vdf_dict, trange, TH=60, Lmax=12, N2D=None, p=3, spline_mincount=2, count_mask=5, ITERATE=False, CREDENTIALS=None, CLIP=False):
        self.TH = TH  
        self.Lmax = Lmax
        self.N2D = N2D
        self.p = p
        self.count_mask = count_mask 
        self.spline_mincount = spline_mincount
        self.ITERATE = ITERATE

        # loading the Slepians tapers once
        self.Slep = eval_Slepians.Slep_transverse()
        self.Slep.gen_Slep_tapers(self.TH, self.Lmax)

        # truncating beyond Shannon number
        if self.N2D is None:
            self.N2D = int(np.sum(self.Slep.V))

        # obtaining the grid points from an actual PSP field-aligned VDF (instrument frame)
        self.setup_timestamp_props(vdf_dict, trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

    def setup_timestamp_props(self, vdf_dict, trange, CREDENTIALS=None, CLIP=False):
        time = vdf_dict.time.data
        energy = vdf_dict.energy.data
        theta = vdf_dict.theta.data
        phi = vdf_dict.phi.data
        vdf = vdf_dict.vdf.data
        count = vdf_dict.counts.data

        # masking the zero count bins where we have no constraints
        vdf[count <= self.count_mask] = np.nan
        vdf[vdf == 0] = np.nan
        self.nanmask = np.isfinite(vdf)

        # get and store the min and maxvalues
        self.minval = np.nanmin(psp_vdf.vdf.data, axis=(1,2,3))
        self.maxval = np.nanmax(psp_vdf.vdf.data, axis=(1,2,3))

        m_p = 0.010438870    # eV/c^2 where c = 299792 km/s
        q_p = 1

        self.velocity = np.sqrt(2 * q_p * energy / m_p)

        # Define the Cartesian Coordinates
        self.vx = self.velocity * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
        self.vy = self.velocity * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
        self.vz = self.velocity * np.sin(np.radians(theta))

        # filemoms = fn.get_psp_span_mom(trange)
        data = fn.init_psp_moms(trange, CREDENTIALS=CREDENTIALS, CLIP=CLIP)

        # obtaining the mangnetic field and v_bulk measured
        self.b_span = data.MAGF_INST.data
        self.v_span = data.VEL_INST.data

        # Get the angle between b and v.
        self.theta_bv = np.degrees(np.arccos(np.einsum('ij, ij->i', self.v_span, self.b_span)/(np.linalg.norm(self.v_span, axis=1) * np.linalg.norm(self.b_span, axis=1))))

        self.l3_time = data.Epoch.data     # check to make sure the span moments match the l2 data!
        self.l2_time = time

    def inversion(self, ubulk, vdfdata, tidx):
        def get_coors(tidx):
            self.vpara, self.vperp1, self.vperp2, self.vperp = None, None, None, None
            self.ubulk = ubulk         # Just to store the data.

            # Shift into the plasma frame
            self.ux = self.vx[tidx] - ubulk[0, NAX, NAX, NAX]
            self.uy = self.vy[tidx] - ubulk[1, NAX, NAX, NAX]
            self.uz = self.vz[tidx] - ubulk[2, NAX, NAX, NAX]

            # Rotate the plasma frame data into the magnetic field aligned frame.
            vpara, vperp1, vperp2 = np.array(fn.rotate_vector_field_aligned(self.ux, self.uy, self.uz,
                                                                            *fn.field_aligned_coordinates(self.b_span[tidx])))
            
            self.vpara, self.vperp1, self.vperp2 = vpara, vperp1, vperp2
            self.vperp = np.sqrt(self.vperp1**2 + self.vperp2**2)

            # # Check angle between flow and magnetic field. 
            if (self.theta_bv[tidx] < 90):
                self.vpara = -1.0 * self.vpara
                self.theta_sign = -1.0
            else: self.theta_sign = 1.0
            # NOTE: NEED TO CHECK ABOVE CALCULATION.
            # self.sign = -1.0*(np.sign(np.median(vpara)))

            # Boosting the vparallel
            self.vshift = np.linalg.norm(self.v_span, axis=1)
            
            self.vpara -= self.vshift[tidx,NAX,NAX,NAX]

            # converting the grid to spherical polar in the field aligned frame
            r, theta, phi = c2s(self.vperp1, self.vperp2, self.vpara)
            self.r_fa = r.value
            self.theta_fa = np.degrees(theta.value) + 90

        def make_knots(tidx):
            self.knots, self.vpara_nonan = None, None

            # finding the minimum and maximum velocities with counts to find the knot locations
            vmin = np.min(self.velocity[tidx, self.nanmask[tidx]])
            vmax = np.max(self.velocity[tidx, self.nanmask[tidx]])
            dlnv = 0.0348
            
            Nbins = int((np.log10(vmax) - np.log10(vmin)) / dlnv)

            # the knot locations
            self.vpara_nonan = self.r_fa[self.nanmask[tidx]] * np.cos(np.radians(self.theta_fa[self.nanmask[tidx]]))
            self.rfac_nonan = self.r_fa[self.nanmask[tidx]]

            counts, bin_edges = np.histogram(np.log10(self.rfac_nonan), bins=Nbins)

            new_edges, _ = merge_bins(bin_edges, counts, self.spline_mincount)
            log_knots = np.sum(new_edges, axis=1)/2

            # discarding knots at counts less than 10 (always discarding the last knot with low count)
            self.knots = np.power(10, log_knots)

            # arranging the knots in an increasing order
            self.knots = np.sort(self.knots)

            # also making the perp grid for future plotting purposes
            self.vperp_nonan = self.r_fa[self.nanmask[tidx]] * np.sin(np.radians(self.theta_fa[self.nanmask[tidx]]))

        def get_Bsplines_scipy():
            t = np.array([self.knots[0] for i in range(self.p)])
            t = np.append(t, self.knots)
            t = np.append(t, np.array([self.knots[-1] for i in range(self.p)]))
            bsp_basis_coefs = np.identity(len(self.knots) + (self.p-1))
            spl = BSpline(t, bsp_basis_coefs, self.p, extrapolate=True)
            self.B_i_n = spl(self.rfac_nonan).T
            self.B_i_n = np.nan_to_num(spl(self.rfac_nonan).T)

        def get_Slepians_scipy():
            self.S_alpha_n = None

            self.theta_nonan = self.theta_fa[self.nanmask[tidx]]

            L = np.arange(0,self.Lmax+1)
            P_scipy = np.asarray([eval_legendre(ell, np.cos(self.theta_nonan * np.pi / 180)) for ell in L])

            # adding the normalization sqrt((2l+1) / 4pi)
            P_scipy = P_scipy * (np.sqrt((2*L + 1) / (4 * np.pi)))[:,NAX]
            S_n_alpha = P_scipy.T @ np.asarray(self.Slep.C)

            # swapping the axes
            self.S_alpha_n = np.moveaxis(S_n_alpha, 0, 1)

            self.S_alpha_n = self.S_alpha_n[:self.N2D,:]

        def get_G_matrix():
            self.G_k_n = None
            self.G_i_alpha_n = None
            # taking the product to make the shape (i x alpha x n)
            self.G_i_alpha_n = self.B_i_n[:,NAX,:] * self.S_alpha_n[NAX,:,:]

            # flattening the k=(i, alpha) dimension to make the shape (k x n)
            npoints = len(self.vpara_nonan)
            self.G_k_n = np.reshape(self.G_i_alpha_n, (-1, npoints))

        def inversion(vdfdata):
            # obtaining the coefficients
            G_g = self.G_k_n @ self.G_k_n.T
            I = np.identity(len(G_g))
            coeffs = np.linalg.inv(G_g + mu * I) @ self.G_k_n @ vdfdata

            # reconstructed VDF (this is the flattened version of the 2D gyrotropic VDF)
            vdf_rec = coeffs @ self.G_k_n

            return vdf_rec

        get_coors(tidx)
        make_knots(tidx)
        get_Bsplines_scipy()
        get_Slepians_scipy()
        get_G_matrix()

        return inversion(vdfdata)

if __name__=='__main__':
    # Initial Parameters
    # trange = ['2020-01-26T07:00:00', '2020-01-26T07:30:00']
    trange = ['2022-02-25T00:00:00', '2022-02-25T01:00:00']
    credentials = fn.load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
    
    # NOTE: Add to separate initialization script in future. 
    TH         = 60
    LMAX       = 12
    N2D        = 3
    P          = 3
    SPLINE_MINCOUNT   = 7
    COUNT_MASK = 1
    ITERATE    = False
    CLIP       = True

    # Load in the VDFs for given timerange
    psp_vdf = fn.init_psp_vdf(trange, CREDENTIALS=creds, CLIP=CLIP)

    # initializing the inversion class
    gvdf_tstamp = gyrovdf(psp_vdf, trange, Lmax=LMAX, TH=TH, N2D=N2D, 
                          count_mask=COUNT_MASK, spline_mincount=SPLINE_MINCOUNT,
                          ITERATE=ITERATE, CREDENTIALS=creds, CLIP=CLIP)

    p0_Slep = []
    p0_Poly = []
    p0_Slep_on_Polyx = []

    Ntimes = len(psp_vdf.time)
    for tidx in tqdm(range(len(psp_vdf.time))):
        bvec = gvdf_tstamp.b_span[tidx] / np.linalg.norm(gvdf_tstamp.b_span[tidx])
        threeD_points = np.vstack([gvdf_tstamp.vx[tidx][gvdf_tstamp.nanmask[tidx]],
                                gvdf_tstamp.vy[tidx][gvdf_tstamp.nanmask[tidx]],
                                gvdf_tstamp.vz[tidx][gvdf_tstamp.nanmask[tidx]]]).T
        threeD_values = psp_vdf.vdf.data[tidx][gvdf_tstamp.nanmask[tidx]]

        # using orgin from the vspan
        origin = gvdf_tstamp.v_span[tidx]

        # using the Slepian loss function
        loss_fn = loss_fn_Slepians
        best_p0_Slep, res_fun = find_symmetry_point(threeD_points, threeD_values, bvec, loss_fn, origin=origin)
        p0_Slep.append(best_p0_Slep)

        # using the Polynomial function
        loss_fn = loss_fn_Polynomials
        best_p0_Poly, res_fun = find_symmetry_point(threeD_points, threeD_values, bvec, loss_fn, origin=origin)    
        p0_Poly.append(best_p0_Poly)

        p0_Slep_on_Polyx.append(point_on_axis_with_x(best_p0_Slep, bvec, gvdf_tstamp.v_span[tidx,0]))

    p0_Slep = np.asarray(p0_Slep)
    p0_Poly = np.asarray(p0_Poly)
    p0_Slep_on_Polyx = np.asarray(p0_Slep_on_Polyx)

    # comparing the centroids
    plt.figure()

    # plotting the x-centroid
    plt.plot(p0_Poly[:,0], 'r', lw=4, alpha=0.5)
    plt.plot(p0_Slep[:,0], 'r')

    # plotting the y-centroid
    plt.plot(p0_Poly[:,1], 'k', lw=4, alpha=0.5)
    plt.plot(p0_Slep[:,1], 'k')

    # plotting the z-centroid
    plt.plot(p0_Poly[:,2], 'b', lw=4, alpha=0.5)
    plt.plot(p0_Slep[:,2], 'b')

    # comparing the centroids for the shifted points
    plt.figure()

    # plotting the x-centroid
    plt.plot(p0_Poly[:,0], 'r', lw=4, alpha=0.5)
    plt.plot(p0_Slep_on_Polyx[:,0], 'r')

    # plotting the y-centroid
    plt.plot(p0_Poly[:,1], 'k', lw=4, alpha=0.5)
    plt.plot(p0_Slep_on_Polyx[:,1], 'k')

    # plotting the z-centroid
    plt.plot(p0_Poly[:,2], 'b', lw=4, alpha=0.5)
    plt.plot(p0_Slep_on_Polyx[:,2], 'b')

    # comparing the centroids for the shifted points
    plt.figure()

    # plotting the x-centroid
    plt.plot(gvdf_tstamp.v_span[:Ntimes,0], 'r', alpha=0.5)
    plt.plot(p0_Slep_on_Polyx[:,0], 'r')

    # plotting the y-centroid
    plt.plot(gvdf_tstamp.v_span[:Ntimes,1], 'k', alpha=0.5)
    plt.plot(p0_Slep_on_Polyx[:,1], 'k')

    # plotting the z-centroid
    plt.plot(gvdf_tstamp.v_span[:Ntimes,2], 'b', alpha=0.5)
    plt.plot(p0_Slep_on_Polyx[:,2], 'b')
