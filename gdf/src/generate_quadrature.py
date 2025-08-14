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

from gdf.src import functions as fn
from gdf.src import misc_funcs as misc_fn

def _gl_nodes_weights_on_interval(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map n-point Gauss-Legendre nodes/weights from [-1,1] to [a,b].
    Returns nodes (shape (n,)) and weights (shape (n,)).
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    x, w = np.polynomial.legendre.leggauss(n)  # [-1,1]
    # Affine map to [a,b]
    xm = 0.5*(b-a)*x + 0.5*(b+a)
    wm = 0.5*(b-a)*w
    return xm, wm

def _midpoint_nodes_weights_on_interval(a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    n-point uniform midpoint/trapezoid hybrid on [a,b].
    If n==1: pure midpoint. If n>1: n equally spaced points with equal weights (trapezoid-like).
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([(a+b)/2.0]), np.array([b-a])
    # uniform nodes
    nodes = np.linspace(a, b, n, endpoint=False) + (b-a)/(2*n)
    weights = np.full(n, (b-a)/n)
    return nodes, weights

def build_cell_quadrature(
    v_edges: ArrayLike,
    theta_edges: ArrayLike,
    phi_edges: ArrayLike,
    n_v: int = 2,
    n_mu: int = 2,
    n_phi: int = 1,
    phi_rule: Literal["midpoint","gl"] = "midpoint",
) -> Dict[str, np.ndarray]:
    """
    Construct tensor-product quadrature for each instrument spherical cell.
    
    Parameters
    ----------
    v_edges : array-like, shape (Nv+1,)
        Monotone increasing radial speed bin *edges* [v0, v1, ..., vNv]. Units arbitrary.
    theta_edges : array-like, shape (Ntheta+1,)
        Monotone increasing polar-angle bin *edges* in radians, in [0, pi].
    phi_edges : array-like, shape (Nphi+1,)
        Monotone increasing azimuthal bin *edges* in radians, typically in [-pi, pi) or [0, 2pi).
    n_v : int
        Number of Gauss–Legendre points in v per cell (default 2).
    n_mu : int
        Number of Gauss–Legendre points in mu=cos(theta) per cell (default 2).
    n_phi : int
        Number of points in phi per cell (default 1).
    phi_rule : {"midpoint","gl"}
        Rule for phi. "midpoint" = 1-pt midpoint when n_phi==1, or uniform n_phi-point trap if n_phi>1.
        "gl" = n_phi-point Gauss–Legendre on [phi_lo, phi_hi].
    
    Returns
    -------
    out : dict of np.ndarray
        Keys:
          - "v":     nodes, shape (Nv, Ntheta, Nphi, Nq)
          - "theta": nodes, shape (Nv, Ntheta, Nphi, Nq)
          - "phi":   nodes, shape (Nv, Ntheta, Nphi, Nq)
          - "w":     weights incl. spherical Jacobian v^2, shape (Nv, Ntheta, Nphi, Nq)
          - "Nq":    scalar int, total nodes per cell (= n_v * n_mu * n_phi)
    Notes
    -----
    The weights integrate f(v,theta,phi) over a *cell*:
        ∫_cell f(v,theta,phi) v^2 sinθ dv dθ dφ ≈ sum_q f(v_q, θ_q, φ_q) * w_q
    This routine uses the change of variable μ=cosθ internally so that sinθ dθ = dμ,
    and accounts for the v^2 Jacobian by multiplying into the weights at the nodes.
    """
    v_edges = np.asarray(v_edges, dtype=float).ravel()
    theta_edges = np.asarray(theta_edges, dtype=float).ravel()
    phi_edges = np.asarray(phi_edges, dtype=float).ravel()
    
    if not (np.all(np.diff(v_edges) > 0) and np.all(np.diff(theta_edges) > 0)):
        raise ValueError("Edges must be strictly increasing for v, theta")

    if not (np.all(np.diff(phi_edges) < 0)):
        raise ValueError("Edges must be strictly decreasing for phi")
    
    Nv = v_edges.size - 1
    Nt = theta_edges.size - 1
    Np = phi_edges.size - 1
    if Nv <= 0 or Nt <= 0 or Np <= 0:
        raise ValueError("Each dimension must have at least one cell (len(edges) >= 2).")
    
    # Prepare storage
    Nq = n_v * n_mu * n_phi
    v_nodes_all   = np.empty((Nv, Nt, Np, Nq), dtype=float)
    th_nodes_all  = np.empty((Nv, Nt, Np, Nq), dtype=float)
    phi_nodes_all = np.empty((Nv, Nt, Np, Nq), dtype=float)
    w_all         = np.empty((Nv, Nt, Np, Nq), dtype=float)
    
    # Precompute 1D reference nodes/weights on each cell edge pair as needed
    # We'll loop over cells to keep the code clear and robust to nonuniform bins.
    q_idx = np.arange(Nq)
    
    for iv in range(Nv):
        v_lo, v_hi = v_edges[iv], v_edges[iv+1]
        v_nodes_1d, v_w_1d = _gl_nodes_weights_on_interval(v_lo, v_hi, n_v)
        for it in range(Nt):
            th_lo, th_hi = theta_edges[it], theta_edges[it+1]
            # Work in μ = cosθ; careful with ordering since μ decreases with θ
            mu_lo_raw, mu_hi_raw = np.cos(th_hi), np.cos(th_lo)  # this ensures mu_lo < mu_hi
            mu_lo, mu_hi = (mu_lo_raw, mu_hi_raw) if mu_hi_raw > mu_lo_raw else (mu_hi_raw, mu_lo_raw)
            mu_nodes_1d, mu_w_1d = _gl_nodes_weights_on_interval(mu_lo, mu_hi, n_mu)
            # Map μ nodes back to θ nodes
            th_nodes_1d = np.arccos(mu_nodes_1d)
            # The change-of-variable sinθ dθ = dμ is already handled by integrating in μ with GL,
            # so no extra sinθ factor goes into the weight from the θ part.
            for ip in range(Np):
                # phi goes from large to small so this is flipped convention
                ph_lo, ph_hi = phi_edges[ip+1], phi_edges[ip]
                if phi_rule == "midpoint":
                    phi_nodes_1d, phi_w_1d = _midpoint_nodes_weights_on_interval(ph_lo, ph_hi, n_phi)
                elif phi_rule == "gl":
                    phi_nodes_1d, phi_w_1d = _gl_nodes_weights_on_interval(ph_lo, ph_hi, n_phi)
                else:
                    raise ValueError("phi_rule must be 'midpoint' or 'gl'")
                
                # Build the tensor product nodes/weights in (v, theta, phi)
                # Order q index as (ivnode, imunode, iphinode) nested loops
                q = 0
                for i_v in range(n_v):
                    for i_mu in range(n_mu):
                        for i_phi in range(n_phi):
                            vq   = v_nodes_1d[i_v]
                            thq  = th_nodes_1d[i_mu]
                            phq  = phi_nodes_1d[i_phi]
                            wq   = (v_w_1d[i_v] * mu_w_1d[i_mu] * phi_w_1d[i_phi]) * (vq**2)  # include Jacobian v^2
                            
                            v_nodes_all[iv, it, ip, q]   = vq
                            th_nodes_all[iv, it, ip, q]  = thq
                            phi_nodes_all[iv, it, ip, q] = phq
                            w_all[iv, it, ip, q]         = wq
                            q += 1
    
    return {"v": v_nodes_all, "theta": th_nodes_all, "phi": phi_nodes_all, "w": w_all, "Nq": Nq}

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
