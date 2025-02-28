#!/usr/bin/env python3
"""
dvripe_physics.py

A more complete DV-RIPE PDE suite. Implements:
  1) PDE initialization (initialize_field).
  2) Operators for each PDE term:
       L_operator     (linear diffusion / wave propagation)
       N_operator     (nonlinear potential)
       G_operator     (gauge interaction)
       G_grav_operator(gravitational coupling)
       M_operator     (membrane coupling)
  3) PDE time-step (run_pde_step) combining all terms.
  4) Post-processing routines:
       detect_vortex_center
       compute_energy_flux
       compute_charge
       compute_spin
       dimensionless_to_keV

Replace placeholders with your real DV-RIPE logic, especially in the PDE step.
"""

import numpy as np

# --------------------------------------------------------------------
# 1) PDE Initialization
# --------------------------------------------------------------------
def initialize_field(params: dict, Nx: int, Ny: int) -> np.ndarray:
    """
    Initialize the DV-RIPE PDE field of shape (Nx, Ny).
    Example: uniform field + small random noise. 
    If your PDE is complex, consider using complex64 arrays.

    Parameters
    ----------
    params : dict
        PDE parameters (gamma, eta, e_gauge, etc.) if needed for init.
    Nx, Ny : int
        Domain size in x and y (or row/col).

    Returns
    -------
    field : np.ndarray
        2D array representing the mass-energy field.
    """
    field = np.ones((Nx, Ny), dtype=np.float32)
    noise = 0.01 * np.random.randn(Nx, Ny).astype(np.float32)
    field += noise
    return field

# --------------------------------------------------------------------
# 2) PDE Operators
# --------------------------------------------------------------------
def L_operator(phi: np.ndarray, params: dict) -> np.ndarray:
    """
    Linear operator: e.g., anisotropic diffusion or wave propagation.

    Example:
      L[phi] = D_r * ∇²(phi)  (basic Laplacian)
    If you want anisotropic or wave terms, adapt accordingly.

    Returns a 2D array of the same shape as phi.
    """
    D_r = params.get("D_r", 0.01)  # example diffusion coefficient
    gx, gy = np.gradient(phi)
    gxx, _  = np.gradient(gx)
    _, gyy  = np.gradient(gy)
    laplacian = gxx + gyy
    return D_r * laplacian

def N_operator(phi: np.ndarray, params: dict) -> np.ndarray:
    """
    Nonlinear potential operator:
      N[phi] = (1/2) lambda_e (|phi|^2 - v_e^2) phi
               + delta_e (phi - phi_other)
               + (1/2) alpha eta (|phi|^4) phi

    This is a placeholder for the typical DV-RIPE nonlinearity.
    If phi is complex, consider |phi|^2 as (phi.real^2 + phi.imag^2).
    """
    lambda_e = params.get("lambda_e", 1.0)
    v_e      = params.get("v_e", 1.0)
    delta_e  = params.get("delta_e", 0.0)
    alpha    = params.get("alpha", 1.0)
    eta      = params.get("eta", 0.5)

    # If phi is complex, define magnitude^2 carefully
    mag_sq = (phi * phi.conjugate()).real if np.iscomplexobj(phi) else phi**2

    # (1/2) lambda_e (|phi|^2 - v_e^2) phi
    term1 = 0.5 * lambda_e * (mag_sq - v_e**2) * phi

    # delta_e (phi - phi_other): We'll do a naive local average for phi_other
    # or set it to zero if not used. 
    # This is a placeholder; adapt as needed.
    phi_other = np.mean(phi)
    term2 = delta_e * (phi - phi_other)

    # (1/2) alpha eta (|phi|^4) phi
    term3 = 0.5 * alpha * eta * (mag_sq**2) * phi

    return term1 + term2 + term3

def G_operator(phi: np.ndarray, params: dict,
               gauge_Ax: np.ndarray=None, gauge_Ay: np.ndarray=None) -> np.ndarray:
    """
    Gauge interaction operator: e.g., non-Abelian SU(2) or simpler U(1).

    For demonstration, let's do a naive U(1) approach:
      G[phi] = - sum_{mu} (D_mu D^mu) phi
    where D_mu phi = ∂_mu phi - i*g*A_mu * phi (if complex).
    If real, or no gauge fields, returns zero array.
    """
    if gauge_Ax is None or gauge_Ay is None:
        return np.zeros_like(phi)

    # If real field, skip
    if not np.iscomplexobj(phi):
        return np.zeros_like(phi)

    # gauge coupling
    g = params.get("e_gauge", 0.1)

    # Covariant derivatives
    gx, gy = np.gradient(phi)
    # D_x = gx - i*g*gauge_Ax*phi
    D_x = gx - 1j*g*gauge_Ax*phi
    D_y = gy - 1j*g*gauge_Ay*phi

    # Now we need partial derivatives of D_x, D_y to form D_mu D^mu
    Dxx, _  = np.gradient(D_x)
    _, Dyy  = np.gradient(D_y)
    # Summation => D_xx + D_yy
    result = Dxx + Dyy
    return -result  # minus sign from eqn

def G_grav_operator(phi: np.ndarray, params: dict,
                    grav_potential: np.ndarray=None) -> np.ndarray:
    """
    Gravitational coupling operator:
      G_grav[phi] = - beta * Phi * phi
    If no grav_potential is provided, returns zero.
    """
    if grav_potential is None:
        return np.zeros_like(phi)
    beta = params.get("beta", 0.01)
    return -beta * grav_potential * phi

def M_operator(phi: np.ndarray, params: dict) -> np.ndarray:
    """
    Membrane coupling operator:
      M[phi] = kappa * <phi>_{boundary}
    Placeholder: if we interpret the boundary average, or some 3D membrane effect.

    Returns an array the same shape as phi, for demonstration we do a uniform shift.
    """
    kappa = params.get("kappa", 0.0)
    if kappa == 0.0:
        return np.zeros_like(phi)

    # Example: add kappa times (mean boundary - phi)
    # We'll do a naive boundary average
    boundary_vals = []
    boundary_vals.extend(phi[0,:])   # top row
    boundary_vals.extend(phi[-1,:])  # bottom row
    boundary_vals.extend(phi[:,0])   # left col
    boundary_vals.extend(phi[:,-1])  # right col
    boundary_avg = np.mean(boundary_vals)

    return kappa * (boundary_avg - phi)

# --------------------------------------------------------------------
# PDE Time-Step
# --------------------------------------------------------------------
def run_pde_step(phi: np.ndarray, dt: float, params: dict,
                 gauge_Ax: np.ndarray=None, gauge_Ay: np.ndarray=None,
                 grav_potential: np.ndarray=None) -> np.ndarray:
    """
    Advance the PDE field by one time step dt using a naive explicit Euler approach:
      phi_new = phi + dt * [ L_operator + N_operator - gamma*phi + G_operator + G_grav_operator + M_operator ].

    For a real production code, you'd likely use an implicit or semi-implicit scheme.
    """
    gamma = params.get("gamma", 0.1)

    # Sum up PDE terms
    lhs_L = L_operator(phi, params)
    lhs_N = N_operator(phi, params)
    lhs_G = G_operator(phi, params, gauge_Ax, gauge_Ay)
    lhs_grav = G_grav_operator(phi, params, grav_potential)
    lhs_M = M_operator(phi, params)

    # PDE eqn => dphi/dt = ...
    rhs = lhs_L + lhs_N - gamma*phi + lhs_G + lhs_grav + lhs_M

    # Update field (naive explicit Euler)
    phi_new = phi + dt * rhs
    return phi_new

# --------------------------------------------------------------------
# Post-Processing Routines
# --------------------------------------------------------------------
def detect_vortex_center(phi: np.ndarray, method: str = "grad_min") -> tuple[int,int]:
    """
    Detect the approximate (row, col) of the vortex nexus in phi, using different methods.

    Parameters
    ----------
    phi : np.ndarray
        2D field array, real or complex.
    method : str, optional
        "grad_min" => find the minimum of |∇phi|.
        "local_extreme" => find the global maximum of phi.
        You can add more strategies as needed.

    Returns
    -------
    (row, col) : tuple[int, int]
        Indices of the detected vortex center.
    """
    if method == "grad_min":
        # Strategy: locate the minimum of |∇phi|
        gx, gy = np.gradient(phi.real if np.iscomplexobj(phi) else phi)
        grad_mag = np.sqrt(gx**2 + gy**2)
        idx_min = np.argmin(grad_mag)
        return np.unravel_index(idx_min, grad_mag.shape)

    elif method == "local_extreme":
        # Strategy: find the global maximum of phi
        # (or use np.argmin if you want a minimum).
        idx_extreme = np.argmax(phi.real if np.iscomplexobj(phi) else phi)
        return np.unravel_index(idx_extreme, phi.shape)

    else:
        raise ValueError(f"Unknown vortex detection method: {method}")

def compute_energy_flux(phi: np.ndarray,
                        gauge_Ax: np.ndarray=None,
                        gauge_Ay: np.ndarray=None,
                        alpha: float=1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Local 2D energy flux. If gauge fields exist and phi is complex, do covariant derivative.
    Otherwise, flux = alpha * phi * ∇phi.
    """
    gx, gy = np.gradient(phi)
    if gauge_Ax is not None and gauge_Ay is not None and np.iscomplexobj(phi):
        # naive U(1) covariant approach
        g = alpha  # or from params
        D_x = gx - 1j*g*gauge_Ax*phi
        D_y = gy - 1j*g*gauge_Ay*phi
        Sx = phi * D_x
        Sy = phi * D_y
    else:
        Sx = alpha * phi * gx
        Sy = alpha * phi * gy
    return (Sx, Sy)

def compute_charge(phi: np.ndarray,
                   gauge_Ax: np.ndarray=None,
                   gauge_Ay: np.ndarray=None) -> float:
    """
    Heuristic "net charge" by integrating gauge flux or topological winding.
    If no gauge fields, returns 0.
    """
    if gauge_Ax is None or gauge_Ay is None:
        return 0.0
    dAy_dx, dAy_dy = np.gradient(gauge_Ay)
    dAx_dx, dAx_dy = np.gradient(gauge_Ax)
    curlA = dAx_dy - dAy_dx
    return float(np.sum(curlA))

def compute_spin(phi: np.ndarray, center: tuple[int,int]) -> float:
    """
    Approximates vortex spin by measuring phase winding around 'center'.
    Requires a complex field. If real, returns 0.0.
    """
    if not np.iscomplexobj(phi):
        return 0.0
    rows, cols = phi.shape
    row_c, col_c = center
    radius = min(rows, cols)//4
    N = 100
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    total_phase_change = 0.0
    phase_prev = None
    for theta in angles:
        rr = row_c + int(radius*np.cos(theta))
        cc = col_c + int(radius*np.sin(theta))
        rr = max(0, min(rows-1, rr))
        cc = max(0, min(cols-1, cc))
        val = phi[rr, cc]
        cur_phase = np.angle(val)
        if phase_prev is not None:
            delta = cur_phase - phase_prev
            delta = (delta + np.pi) % (2*np.pi) - np.pi
            total_phase_change += delta
        phase_prev = cur_phase
    return total_phase_change / (2.0*np.pi)

def dimensionless_to_keV(energy_dimless: float, base_keV: float=511.0) -> float:
    """
    Converts dimensionless energy units to keV.
    By default, 1.0 dimensionless => 511 keV (electron rest mass).
    """
    return energy_dimless * base_keV
