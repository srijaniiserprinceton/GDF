# We'll implement a function that builds per-cell tensor-product quadrature nodes/weights
# for an instrument spherical grid defined by *edges* in v, theta, phi.
#
# Output arrays have shape (Nv, Ntheta, Nphi, Nq), where Nq = n_v * n_mu * n_phi.
# The weights include the spherical Jacobian v^2, i.e., integrate f(v,theta,phi) over the *cell* via:
#   integral_cell ≈ sum_q f(v[q], theta[q], phi[q]) * weights[q]
#
# We use Gauss–Legendre in v and mu=cos(theta) (constant weights),
# and either midpoint or Gauss–Legendre in phi (configurable).
#
# Also included: a tiny demo at the bottom to verify shapes.
from typing import Literal, Tuple, Dict
import numpy as np
from numpy.typing import ArrayLike
import sys, importlib

from gdf.src_GL import functions as fn
from gdf.src_GL import misc_funcs as misc_fn

def _gl_nodes_weights_on_interval(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    x, w = np.polynomial.legendre.leggauss(n)  # [-1,1]
    xm = 0.5*(b-a)*x + 0.5*(b+a)
    wm = 0.5*(b-a)*w
    return xm, wm

def _midpoint_nodes_weights_on_interval(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([(a+b)/2.0]), np.array([b-a])
    nodes = np.linspace(a, b, n, endpoint=False) + (b-a)/(2*n)
    weights = np.full(n, (b-a)/n)
    return nodes, weights

def build_cell_quadrature(
    v_edges: ArrayLike,
    theta_edges: ArrayLike,   # deg, strictly increasing in (-90, 90)
    phi_edges: ArrayLike,     # deg, strictly decreasing (e.g., 180 -> 0)
    n_v: int = 2,
    n_mu: int = 2,            # GL points in μ = sin(theta)
    n_phi: int = 1,
    phi_rule: Literal["midpoint","gl"] = "midpoint"
) -> Dict[str, np.ndarray]:
    """
    Construct tensor-product quadrature for each instrument spherical cell.

    Inputs
    ------
    v_edges : increasing
    theta_edges : degrees, strictly increasing in (-90, 90)  [latitude-like θ]
    phi_edges : degrees, strictly DECREASING (e.g., 180 -> 0)

    Integral & change of variables
    ------------------------------
    ∫ f(v,θ,φ) v^2 cosθ dv dθ dφ, with μ = sinθ  ⇒  cosθ dθ = dμ.
    φ quadrature is built in *radians measure* (weights in radians).

    Returns
    -------
    dict with arrays of shape (Nv, Ntheta, Nphi, Nq):
      "v"     : node speeds
      "theta" : node latitudes in degrees
      "phi"   : node azimuths (degrees if return_phi_in_degrees=True else radians)
      "w"     : weights including v^2 and angular measures (φ in radians)
      "Nq"    : n_v * n_mu * n_phi
      plus echoed edge arrays.
    """
    v_edges = np.asarray(v_edges, dtype=float).ravel()
    theta_edges = np.asarray(theta_edges, dtype=float).ravel()  # deg
    phi_edges = np.asarray(phi_edges, dtype=float).ravel()      # deg

    if not (np.all(np.diff(v_edges) > 0) and np.all(np.diff(theta_edges) > 0)):
        raise ValueError("Edges must be strictly increasing for v, theta")
    if not np.all(np.diff(phi_edges) < 0):
        raise ValueError("Edges must be strictly decreasing for phi")

    Nv = v_edges.size - 1
    Nt = theta_edges.size - 1
    Np = phi_edges.size - 1
    if Nv <= 0 or Nt <= 0 or Np <= 0:
        raise ValueError("Each dimension must have at least one cell (len(edges) >= 2).")

    Nq = n_v * n_mu * n_phi
    v_nodes_all   = np.empty((Nv, Nt, Np, Nq), dtype=float)
    th_nodes_all  = np.empty((Nv, Nt, Np, Nq), dtype=float)  # deg
    phi_nodes_all = np.empty((Nv, Nt, Np, Nq), dtype=float)  # deg or rad
    w_all         = np.empty((Nv, Nt, Np, Nq), dtype=float)

    for iv in range(Nv):
        v_lo, v_hi = v_edges[iv], v_edges[iv+1]
        v_nodes_1d, v_w_1d = _gl_nodes_weights_on_interval(v_lo, v_hi, n_v)

        for it in range(Nt):
            th_lo, th_hi = theta_edges[it], theta_edges[it+1]

            # μ = sinθ with θ in radians (monotone on (-90°, 90°))
            mu_lo = np.sin(np.radians(th_lo))
            mu_hi = np.sin(np.radians(th_hi))
            mu_nodes_1d, mu_w_1d = _gl_nodes_weights_on_interval(mu_lo, mu_hi, n_mu)

            # Safety: clip then map back to θ in degrees via arcsin
            mu_nodes_1d = np.clip(mu_nodes_1d, -1.0, 1.0)
            th_nodes_1d_rad = np.arcsin(mu_nodes_1d)
            th_nodes_1d_deg = np.degrees(th_nodes_1d_rad)

            for ip in range(Np):
                # phi edges are decreasing; build [lo, hi] in radians (lo < hi)
                ph_lo = np.radians(phi_edges[ip+1])
                ph_hi = np.radians(phi_edges[ip])

                if phi_rule == "midpoint":
                    phi_nodes_1d_rad, phi_w_1d = _midpoint_nodes_weights_on_interval(ph_lo, ph_hi, n_phi)
                elif phi_rule == "gl":
                    phi_nodes_1d_rad, phi_w_1d = _gl_nodes_weights_on_interval(ph_lo, ph_hi, n_phi)
                else:
                    raise ValueError("phi_rule must be 'midpoint' or 'gl'")

                q = 0
                for i_v in range(n_v):
                    for i_mu in range(n_mu):
                        for i_phi in range(n_phi):
                            vq   = v_nodes_1d[i_v]
                            thqD = th_nodes_1d_deg[i_mu]                 # deg
                            phqR = phi_nodes_1d_rad[i_phi]               # rad
                            wq   = (v_w_1d[i_v] * mu_w_1d[i_mu] * phi_w_1d[i_phi]) * (vq**2)

                            v_nodes_all[iv, it, ip, q]   = vq
                            th_nodes_all[iv, it, ip, q]  = thqD
                            phi_nodes_all[iv, it, ip, q] = np.degrees(phqR)
                            w_all[iv, it, ip, q] = wq
                            q += 1

    return {
        "v": v_nodes_all,
        "theta": th_nodes_all,   # deg
        "phi": phi_nodes_all,    # deg
        "w": w_all,              # weights in radian measure
        "Nq": Nq,
        "v_edges": v_edges,
        "theta_edges": theta_edges,
        "phi_edges": phi_edges,
    }

def edges_from_centers(x):
    dx = np.diff(x)
    edge = np.empty(len(x) + 1)
    edge[1:-1] = x[:-1] + dx/2
    edge[0]    = x[0]    - dx[0]/2
    edge[-1]   = x[-1]   + dx[-1]/2
    return edge

def generate_grid_edges(vdf_dict, tidx):
    E = vdf_dict.energy[tidx,:,0,0]

    theta = vdf_dict.theta[tidx,0,:,0]
    phi = vdf_dict.phi[tidx,0,0,:]

    E_edges = np.power(10, edges_from_centers(np.log10(E)))
    T_edges = edges_from_centers(theta)
    P_edges = edges_from_centers(phi)

    # converting E_edges to v_edges
    q_p, m_p = 1, 0.010438870
    V_edges = np.sqrt(2 * q_p * E_edges / m_p)

    return V_edges, T_edges, P_edges

def GL_vol_avg_polcap(G_k_n_nq_GL, gvdf_tstamp, tidx):
    # exctracting the weight based on the method (cartesian needs a reflected grid and polcap does not)
    w = gvdf_tstamp.w_nonan_GL

    # Sum over the quadrature dimension (last axis) to integrate each cell in the instrument frame
    I_VOL = np.sum(G_k_n_nq_GL * w[np.newaxis, :, :], axis=-1)

    # calculating the volume in each element
    dv_term = (gvdf_tstamp.GL["v_edges"][1:]**3 - gvdf_tstamp.GL["v_edges"][:-1]**3) / 3.0                   # (Nv,)
    dmu     = np.sin(np.radians(gvdf_tstamp.GL["theta_edges"][1:])) - np.sin(np.radians(gvdf_tstamp.GL["theta_edges"][:-1]))           # (Nt,)
    dphi    = (np.radians(gvdf_tstamp.GL["phi_edges"][:-1]) - np.radians(gvdf_tstamp.GL["phi_edges"][1:]))                           # (Np,)

    # Broadcast to (Nv,Nt,Np)
    VOL = (dv_term[:,None,None] * dmu[None,:,None] * dphi[None,None,:])[gvdf_tstamp.nanmask[tidx]]           # (alpha, ninst_grids)
    
    return I_VOL / VOL

def GL_vol_avg_cartesian(G, gvdf_tstamp, tidx):
    # exctracting the weight based on the method (cartesian needs a reflected grid and polcap does not)
    wflat = np.reshape(gvdf_tstamp.w_nonan_GL, (-1), 'C')
    w = np.concatenate([wflat, wflat])
    Gw = G * w[:,np.newaxis]

    Nsleps = G.shape[-1] 
    Npoints = gvdf_tstamp.v_para_all.shape[0]
    NQ = gvdf_tstamp.NQ_V * gvdf_tstamp.NQ_T * gvdf_tstamp.NQ_P

    # Sum over the quadrature dimension to integrate each cell in the instrument frame
    I_VOL = np.sum(np.reshape(Gw, (Npoints, NQ, Nsleps)), axis=1)

    # calculating the volume in each element
    dv_term = (gvdf_tstamp.GL["v_edges"][1:]**3 - gvdf_tstamp.GL["v_edges"][:-1]**3) / 3.0                   # (Nv,)
    dmu     = np.sin(np.radians(gvdf_tstamp.GL["theta_edges"][1:])) - np.sin(np.radians(gvdf_tstamp.GL["theta_edges"][:-1]))           # (Nt,)
    dphi    = (np.radians(gvdf_tstamp.GL["phi_edges"][:-1]) - np.radians(gvdf_tstamp.GL["phi_edges"][1:]))                           # (Np,)

    # Broadcast to (Nv,Nt,Np)
    VOL = (dv_term[:,None,None] * dmu[None,:,None] * dphi[None,None,:])[gvdf_tstamp.nanmask[tidx]]           # (ninst_grids,)
    VOL = np.concatenate([VOL, VOL])
    
    return I_VOL / VOL[:,np.newaxis]

# --- Tiny demo to verify shapes ---
if __name__ == "__main__":
    # loading the psp interval
    sys.path.append('/Users/sbdas/Documents/Research/SpaceWeather/GDF')
    config_file = 'init_gdf'
    config = importlib.import_module(config_file).config

    # loading the credentials from the file
    creds  = misc_fn.credential_reader(None) #config['global']['CREDS_PATH'])

    # loading the PSP data for the given TRANGE with optional clipping
    psp_vdf = fn.init_psp_vdf(config['global']['TRANGE'], CREDENTIALS=creds, CLIP=config['global']['CLIP'])

    # computing the edges
    V_edges, T_edges, P_edges = generate_grid_edges(psp_vdf, 0)
    
    out = build_cell_quadrature(V_edges, T_edges, P_edges, n_v=2, n_mu=2, n_phi=2, phi_rule="midpoint")
    print("Nq =", out["Nq"])
    print("v nodes shape:", out["v"].shape)
    print("theta nodes shape:", out["theta"].shape)
    print("phi nodes shape:", out["phi"].shape)
    print("weights shape:", out["w"].shape)
