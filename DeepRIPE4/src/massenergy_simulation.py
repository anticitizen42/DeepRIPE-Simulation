#!/usr/bin/env python3
"""
src/massenergy_simulation.py
Version 1.6

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
progress monitoring printed during integration.
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
Nr = 128              # radial grid points
Ntheta = 256          # angular grid points
r_max = 50.0          # maximum radius
r = np.linspace(0, r_max, Nr)
theta = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
dr = r[1]-r[0]
dtheta = theta[1]-theta[0]

FINAL_TIME = 1.0
t0 = 0.0
DT_INIT = 0.005
DIAG_INTERVAL = 0.05

# -----------------
# PDE Parameters (Theory Workshop)
# -----------------
params = {
    'D_r': 1.0,             # radial diffusion
    'D_theta': 0.8,         # angular diffusion
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
    phi_r[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dr)
    phi_r[0, :] = (phi[1, :] - phi[0, :]) / dr
    phi_r[-1, :] = (phi[-1, :] - phi[-2, :]) / dr
    phi_rr[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / (dr**2)
    phi_rr[0, :] = (phi[1, :] - 2*phi[0, :] + phi[1, :]) / (dr**2)
    phi_rr[-1, :] = (phi[-2, :] - 2*phi[-1, :] + phi[-2, :]) / (dr**2)
    phi_theta_theta = (np.roll(phi, -1, axis=1) - 2*phi + np.roll(phi, 1, axis=1)) / (dtheta**2)
    r_matrix = np.tile(r.reshape((Nr, 1)), (1, Ntheta))
    with np.errstate(divide='ignore', invalid='ignore'):
        lap = D_r * phi_rr + np.where(r_matrix==0, 0, (1.0/r_matrix)*phi_r) + (D_theta/(r_matrix**2)) * phi_theta_theta
    lap[r_matrix==0] = D_r * phi_rr[r_matrix==0]
    return lap

def nonlinear_potential(phi, params):
    lambda_e = params['lambda_e']
    v_e = params['v_e']
    delta_e = params['delta_e']
    alpha = params['alpha']
    eta = params['eta']
    kernel = np.ones((3,3))/9.0
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
    # Only print if at least 0.1 time units advanced and at least 1 second has elapsed in wall-clock time.
    if (t - _last_reported_t) >= 0.1 and (current_time - _last_progress_print) >= 1.0:
        print(f"Solver progress: simulation time t = {t:.4f}")
        _last_progress_print = current_time
        _last_reported_t = t

# -----------------
# Full PDE Right-Hand Side in Polar Coordinates
# -----------------
def pde_rhs(t, y, params, r, dr, dtheta, Nr, Ntheta):
    # Report progress
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
# Diagnostics Functions (same as previous versions)
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

def cart2polar(field, center=None, final_radius=None, radial_steps=100, angular_steps=360):
    Ny, Nx = field.shape
    if center is None:
        center = (Nx/2, Ny/2)
    if final_radius is None:
        final_radius = np.sqrt(max(center[0], Nx-center[0])**2 + max(center[1], Ny-center[1])**2)
    r_lin = np.linspace(0, final_radius, radial_steps)
    theta_lin = np.linspace(0, 2*np.pi, angular_steps, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_lin, theta_lin, indexing='ij')
    Xc, Yc = center
    X = Xc + r_grid * np.cos(theta_grid)
    Y = Yc + r_grid * np.sin(theta_grid)
    from scipy.ndimage import map_coordinates
    polar_field = map_coordinates(field, [Y.ravel(), X.ravel()], order=1, mode='reflect')
    return r_grid, theta_grid, polar_field.reshape(r_grid.shape)

def extract_zoom(field, center, window_size=10):
    row, col = center
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, field.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, field.shape[1])
    return field[r_start:r_end, c_start:c_end]

def compute_vortex_spin(zoom_region):
    _, _, polar_zoom = cart2polar(zoom_region, radial_steps=100, angular_steps=360)
    radial_idx = polar_zoom.shape[0] // 2
    angular_signal = polar_zoom[radial_idx, :] - np.mean(polar_zoom[radial_idx, :])
    analytic_signal = hilbert(angular_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    circulation = phase[-1] - phase[0]
    return circulation / (2*np.pi)

def compute_vortex_energy(zoom_region):
    return np.sum(zoom_region**2) / zoom_region.size

def compute_vortex_charge(zoom_region):
    return -np.sum(zoom_region - np.mean(zoom_region)) / zoom_region.size

def analyze_dual_resonance(polar_field, radial_index=None):
    if radial_index is None:
        radial_index = polar_field.shape[0] // 2
    angular_signal = polar_field[radial_index, :] - np.mean(polar_field[radial_index, :])
    fft_vals = np.fft.rfft(angular_signal)
    freqs = np.fft.rfftfreq(len(angular_signal), d=1)
    amplitudes = np.abs(fft_vals)
    peaks, _ = find_peaks(amplitudes)
    if peaks.size < 2:
        return None
    sorted_indices = np.argsort(amplitudes[peaks])[::-1]
    primary_peaks = peaks[sorted_indices[:2]]
    return [(freqs[p], amplitudes[p]) for p in primary_peaks]

def compute_diagnostics(t_val, y):
    phi = y.reshape((Nr, Ntheta))
    diag = {}
    diag['time'] = t_val
    diag.update(compute_global_statistics(phi))
    diag['autocorrelation_length'] = np.nan  # placeholder
    vortex_center = np.unravel_index(np.argmax(phi), phi.shape)
    diag['vortex_center'] = vortex_center
    zoom_region = extract_zoom(phi, vortex_center, window_size=10)
    diag['vortex_zoom_mean'] = np.mean(zoom_region)
    diag['vortex_zoom_std'] = np.std(zoom_region)
    diag['vortex_spin'] = compute_vortex_spin(zoom_region)
    diag['vortex_energy'] = compute_vortex_energy(zoom_region)
    diag['vortex_charge'] = compute_vortex_charge(zoom_region)
    _, _, polar_zoom = cart2polar(zoom_region, radial_steps=100, angular_steps=360)
    dual_modes = analyze_dual_resonance(polar_zoom)
    diag['dual_modes'] = dual_modes if dual_modes is not None else "None"
    _, _, polar_full = cart2polar(phi, radial_steps=200, angular_steps=360)
    diag['mean_polar'] = np.mean(polar_full)
    diag['std_polar'] = np.std(polar_full)
    return diag

# -----------------
# Main Simulation Routine
# -----------------
def main():
    data_folder = "data"
    snapshots_folder = os.path.join(data_folder, "field_snapshots")
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(snapshots_folder, exist_ok=True)
    
    # Initial field: nearly uniform plus small noise.
    phi0 = np.ones((Nr, Ntheta), dtype=np.float64) + 0.01 * np.random.randn(Nr, Ntheta)
    y0 = phi0.ravel()
    
    t_span = (t0, FINAL_TIME)
    t_eval = np.arange(t0, FINAL_TIME + DIAG_INTERVAL, DIAG_INTERVAL)
    
    sol = solve_ivp(fun=lambda t, y: pde_rhs(t, y, params, r, dr, dtheta, Nr, Ntheta),
                    t_span=t_span, y0=y0, method="Radau", t_eval=t_eval)
    
    if not sol.success:
        print("Solver failed!")
        return
    
    # Save snapshots sequentially.
    for t_val, y in zip(sol.t, sol.y.T):
        phi = y.reshape((Nr, Ntheta))
        np.save(os.path.join(snapshots_folder, f"field_{t_val:.4f}.npy"), phi)
    
    # Parallelize diagnostic computations.
    from concurrent.futures import ProcessPoolExecutor
    diag_args = list(zip(sol.t, sol.y.T))
    with ProcessPoolExecutor() as executor:
        diagnostics_list = list(executor.map(compute_diagnostics, [arg[0] for arg in diag_args],
                                               [arg[1] for arg in diag_args]))
    
    final_file = os.path.join(data_folder, "final_massenergy_field.npy")
    np.save(final_file, sol.y[:, -1].reshape((Nr, Ntheta)))
    diag_filename = os.path.join(data_folder, "extended_diagnostics.npy")
    np.save(diag_filename, np.array(diagnostics_list, dtype=object))
    print("Simulation complete. Extended diagnostics saved.")

if __name__ == "__main__":
    main()
