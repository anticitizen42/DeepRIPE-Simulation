#!/usr/bin/env python3
"""
massenergy_simulation.py
Version 1.7

This script evolves the DV‑RIPE mass–energy field φ on a polar grid
using the full PDE:

    ∂φ/∂t = L[φ] + N[φ] - γ φ + G[φ] + G_grav[φ] + M[φ] + ξ η(t)

with the following operators:

  • L[φ] = D_r φ_rr + (1/r) φ_r + (D_θ/r²) φ_θθ,  (anisotropic Laplacian)
  • N[φ] = ½λₑ (|φ|² − vₑ²) φ + δₑ (φ − ⟨φ⟩_local) + ½αη |φ|⁴ φ,
  • Damping: −γ φ,
  • Gauge:   G[φ] ≈ −e_gauge * (φ(r,θ+Δθ) − φ(r,θ))/Δθ,
  • Grav.:   G_grav[φ] = −β ⟨|φ|²⟩ φ,
  • Membrane: M[φ] = κ*(⟨φ⟩_boundary − φ),
  • Jitter:  ξη(t) with white noise.
  
Extended diagnostics are saved in the data folder. This version includes
extensive diagnostics that track the initialization of the field, its collapse
to a point (the vortex nexus), and analysis for dual-resonant structures bilaterally
extending from that nexus.
"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import convolve, zoom
from scipy.signal import hilbert, find_peaks

# Global progress monitor variables
_last_progress_print = time.time()
_last_reported_t = -np.inf

# -----------------
# Grid and Time Settings (Polar Domain)
# -----------------
Nr = 128              # number of radial grid points
Ntheta = 256          # number of angular grid points
r_max = 50.0          # maximum radius in the domain
r = np.linspace(0, r_max, Nr)
theta = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

FINAL_TIME = 1.0      # total simulation time
t0 = 0.0              # initial time
DIAG_INTERVAL = 0.05  # time interval for diagnostics snapshots

# -----------------
# PDE Parameters (from Theory Workshop)
# -----------------
params = {
    'D_r': 1.0,             # radial diffusion coefficient
    'D_theta': 0.8,         # angular diffusion coefficient
    'lambda_e': 1.0,
    'v_e': 1.0,
    'delta_e': 0.1,
    'alpha': 1.0,
    'eta': 0.1,
    'gamma': 0.1,
    'e_gauge': 0.05,
    'beta': 0.0005,
    'kappa': 0.5,
    'xi': 0.001
}

# -----------------
# PDE Operators in Polar Coordinates
# -----------------
def polar_laplacian(phi, r, dr, dtheta, D_r, D_theta):
    Nr, Ntheta = phi.shape
    phi_r = np.zeros_like(phi)
    phi_rr = np.zeros_like(phi)
    # First-order radial derivative
    phi_r[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dr)
    phi_r[0, :] = (phi[1, :] - phi[0, :]) / dr
    phi_r[-1, :] = (phi[-1, :] - phi[-2, :]) / dr
    # Second-order radial derivative
    phi_rr[1:-1, :] = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / (dr**2)
    phi_rr[0, :] = (phi[1, :] - 2 * phi[0, :] + phi[1, :]) / (dr**2)
    phi_rr[-1, :] = (phi[-2, :] - 2 * phi[-1, :] + phi[-2, :]) / (dr**2)
    # Second-order angular derivative (using periodic boundaries)
    phi_theta_theta = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / (dtheta**2)
    # Build a radial matrix for division (avoiding division by zero)
    r_matrix = np.tile(r.reshape((Nr, 1)), (1, Ntheta))
    with np.errstate(divide='ignore', invalid='ignore'):
        lap = D_r * phi_rr + np.where(r_matrix == 0, 0, (1.0 / r_matrix) * phi_r) \
              + (D_theta / (r_matrix**2)) * phi_theta_theta
    lap[r_matrix == 0] = D_r * phi_rr[r_matrix == 0]
    return lap

def nonlinear_potential(phi, params):
    lambda_e = params['lambda_e']
    v_e = params['v_e']
    delta_e = params['delta_e']
    alpha = params['alpha']
    eta = params['eta']
    kernel = np.ones((3, 3)) / 9.0
    local_mean = convolve(phi, kernel, mode='reflect')
    term1 = 0.5 * lambda_e * ((phi**2) - v_e**2) * phi
    term2 = delta_e * (phi - local_mean)
    term3 = 0.5 * alpha * eta * (phi**4) * phi
    return term1 + term2 + term3

def damping_term(phi, params):
    return -params['gamma'] * phi

def gauge_interaction(phi, params, dtheta):
    e_gauge = params['e_gauge']
    gauge_diff = (np.roll(phi, -1, axis=1) - phi) / dtheta
    return -e_gauge * gauge_diff

def gravitational_term(phi, params):
    beta = params['beta']
    density = np.mean(phi**2)
    return -beta * density * phi

def membrane_coupling(phi, params):
    kappa = params['kappa']
    boundary_avg = np.mean(phi[-1, :])
    return kappa * (boundary_avg - phi)

def phase_jitter(phi, params):
    xi = params['xi']
    return xi * np.random.randn(*phi.shape)

# -----------------
# Progress Monitor (called within pde_rhs)
# -----------------
def progress_monitor(t):
    global _last_progress_print, _last_reported_t
    current_time = time.time()
    if (t - _last_reported_t) >= 0.1 and (current_time - _last_progress_print) >= 1.0:
        print(f"Solver progress: simulation time t = {t:.4f}")
        _last_progress_print = current_time
        _last_reported_t = t

# -----------------
# Full PDE Right-Hand Side in Polar Coordinates
# -----------------
def pde_rhs(t, y, params, r, dr, dtheta, Nr, Ntheta):
    progress_monitor(t)
    phi = y.reshape((Nr, Ntheta))
    L_phi = polar_laplacian(phi, r, dr, dtheta, params['D_r'], params['D_theta'])
    N_phi = nonlinear_potential(phi, params)
    damp = damping_term(phi, params)
    gauge = gauge_interaction(phi, params, dtheta)
    grav = gravitational_term(phi, params)
    membrane = membrane_coupling(phi, params)
    jitter = phase_jitter(phi, params)
    dphi_dt = L_phi + N_phi + damp + gauge + grav + membrane + jitter
    return dphi_dt.ravel()

# -----------------
# Extended Diagnostics Functions
# -----------------
def compute_global_statistics(phi):
    mean_field = np.mean(phi)
    std_field = np.std(phi)
    grad_y, grad_x = np.gradient(phi)
    mean_gradient = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    max_gradient = np.max(np.sqrt(grad_x**2 + grad_y**2))
    fft_phi = np.fft.fft2(phi)
    spec_energy = np.abs(fft_phi)**2
    return {
        'mean_field': mean_field,
        'std_field': std_field,
        'mean_gradient': mean_gradient,
        'max_gradient': max_gradient,
        'peak_spectral_energy': np.max(spec_energy),
        'total_spectral_energy': np.sum(spec_energy)
    }

def detect_vortex_nexus(phi):
    """
    Detect the vortex nexus, defined as the point of minimum field amplitude.
    Returns the nexus coordinates, the minimum amplitude, and an estimate of the core radius.
    """
    amplitude = np.abs(phi)
    nexus_idx = np.unravel_index(np.argmin(amplitude), amplitude.shape)
    min_ampl = amplitude[nexus_idx]
    # Estimate a core radius by thresholding around the nexus:
    threshold = 1.5 * min_ampl
    mask = amplitude <= threshold
    core_pixels = np.argwhere(mask)
    # Compute an effective radius as the average distance from the nexus.
    distances = np.sqrt((core_pixels[:,0] - nexus_idx[0])**2 + (core_pixels[:,1] - nexus_idx[1])**2)
    core_radius = np.mean(distances) if distances.size > 0 else 0.0
    return {
        'nexus_coords': nexus_idx,
        'min_amplitude': min_ampl,
        'core_radius': core_radius
    }

def dual_resonance_features(phi, nexus_coords, window_size=20):
    """
    Analyze the region surrounding the vortex nexus for dual-resonant features.
    The field is converted to polar coordinates about the nexus, and an FFT/hilbert
    analysis is performed along the angular direction to detect bilateral peaks.
    """
    # Extract a zoomed region around the nexus.
    row, col = nexus_coords
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, phi.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, phi.shape[1])
    zoom_region = phi[r_start:r_end, c_start:c_end]
    
    # Convert the zoom region to polar coordinates about its center.
    center = ((c_end - c_start)/2, (r_end - r_start)/2)
    from scipy.ndimage import map_coordinates
    radial_steps = 100
    angular_steps = 360
    r_lin = np.linspace(0, window_size, radial_steps)
    theta_lin = np.linspace(0, 2*np.pi, angular_steps, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_lin, theta_lin, indexing='ij')
    Xc, Yc = center
    X = Xc + r_grid * np.cos(theta_grid)
    Y = Yc + r_grid * np.sin(theta_grid)
    polar_region = map_coordinates(np.abs(zoom_region), [Y.ravel(), X.ravel()], order=1, mode='reflect')
    polar_region = polar_region.reshape((radial_steps, angular_steps))
    
    # At a fixed radial distance (e.g., halfway), extract the angular signal.
    radial_idx = radial_steps // 2
    angular_signal = polar_region[radial_idx, :] - np.mean(polar_region[radial_idx, :])
    analytic_signal = hilbert(angular_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    circulation = phase[-1] - phase[0]
    
    # FFT to detect bilateral peaks.
    fft_vals = np.fft.rfft(angular_signal)
    freqs = np.fft.rfftfreq(len(angular_signal), d=1)
    amplitudes = np.abs(fft_vals)
    peaks, _ = find_peaks(amplitudes, height=np.max(amplitudes)*0.3)
    
    return {
        'angular_circulation': circulation,
        'fft_peaks': peaks.tolist(),
        'fft_amplitudes': amplitudes.tolist()
    }

def extended_diagnostics(t_val, y):
    """
    Compute an extended set of diagnostics at time t_val.
    This includes global statistics, vortex nexus detection, and dual-resonant features.
    """
    phi = y.reshape((Nr, Ntheta))
    diag = {'time': t_val}
    diag.update(compute_global_statistics(phi))
    
    # Detect vortex nexus.
    nexus_data = detect_vortex_nexus(phi)
    diag.update(nexus_data)
    
    # Analyze dual-resonant features near the nexus.
    dual_data = dual_resonance_features(phi, nexus_data['nexus_coords'])
    diag['dual_resonance'] = dual_data
    
    return diag

# -----------------
# Main Simulation Routine
# -----------------
def main():
    data_folder = "data"
    snapshots_folder = os.path.join(data_folder, "field_snapshots")
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(snapshots_folder, exist_ok=True)
    
    # Initial field: nearly uniform (ones) with small noise.
    phi0 = np.ones((Nr, Ntheta), dtype=np.float64) + 0.01 * np.random.randn(Nr, Ntheta)
    y0 = phi0.ravel()
    
    t_span = (t0, FINAL_TIME)
    t_eval = np.arange(t0, FINAL_TIME + DIAG_INTERVAL, DIAG_INTERVAL)
    
    sol = solve_ivp(
        fun=lambda t, y: pde_rhs(t, y, params, r, dr, dtheta, Nr, Ntheta),
        t_span=t_span,
        y0=y0,
        method="Radau",
        t_eval=t_eval
    )
    
    if not sol.success:
        print("Solver failed!")
        return
    
    # Save field snapshots.
    for t_val, y in zip(sol.t, sol.y.T):
        phi = y.reshape((Nr, Ntheta))
        np.save(os.path.join(snapshots_folder, f"field_{t_val:.4f}.npy"), phi)
    
    # Compute extended diagnostics in parallel.
    from concurrent.futures import ProcessPoolExecutor
    diag_args = list(zip(sol.t, sol.y.T))
    with ProcessPoolExecutor() as executor:
        diagnostics_list = list(executor.map(
            extended_diagnostics,
            [arg[0] for arg in diag_args],
            [arg[1] for arg in diag_args]
        ))
    
    final_file = os.path.join(data_folder, "final_massenergy_field.npy")
    np.save(final_file, sol.y[:, -1].reshape((Nr, Ntheta)))
    diag_filename = os.path.join(data_folder, "extended_diagnostics.npy")
    np.save(diag_filename, np.array(diagnostics_list, dtype=object))
    print("Simulation complete. Extended diagnostics saved.")

if __name__ == "__main__":
    main()
