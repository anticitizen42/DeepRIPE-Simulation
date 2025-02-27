#!/usr/bin/env python3
"""
massenergy_simulation_vulkan.py
Version 1.7-VULKAN

This script evolves the DV‑RIPE mass–energy field φ on a polar grid
using the full PDE:

    ∂φ/∂t = L[φ] + N[φ] - γ φ + G[φ] + G_grav[φ] + M[φ] + ξ η(t)

The heavy PDE operator computations are offloaded to the GPU via Vulkan.
Extended diagnostics are saved in the data folder.

Note: Currently the Vulkan compute module implements a placeholder routine
      (compute_pde_rhs) that must be extended to perform the full operator.
"""

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert, find_peaks

# Import our Vulkan compute helper (assumed to be in vulkan_compute.py)
from vulkan_compute import VulkanCompute

# Global progress monitor variables
_last_progress_print = time.time()
_last_reported_t = -np.inf

# -----------------
# Grid and Time Settings (Polar Domain)
# -----------------
Nr = 128              # number of radial grid points
Ntheta = 256          # number of angular grid points
r_max = 50.0          # maximum radius
r = np.linspace(0, r_max, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]

FINAL_TIME = 1.0      # total simulation time
t0 = 0.0              # initial time
DIAG_INTERVAL = 0.05  # diagnostic interval

# -----------------
# PDE Parameters (Theory Workshop)
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
# Progress Monitor
# -----------------
def progress_monitor(t):
    global _last_progress_print, _last_reported_t
    current_time = time.time()
    if (t - _last_reported_t) >= 0.1 and (current_time - _last_progress_print) >= 1.0:
        print(f"Solver progress: simulation time t = {t:.4f}")
        _last_progress_print = current_time
        _last_reported_t = t

# -----------------
# PDE Right-Hand Side using Vulkan Compute
# -----------------
def pde_rhs_vulkan(t, y, params, grid_shape, vulkan_comp):
    progress_monitor(t)
    # y is a 1D NumPy array representing φ on our grid (flattened).
    # We pass y and additional parameters to our Vulkan compute module.
    #
    # The VulkanCompute instance is expected to have a method
    # compute_pde_rhs(state, params, grid_shape, grid_info)
    # that returns the computed time derivative as a NumPy array.
    #
    # For now, compute_pde_rhs is a placeholder that might, say, scale y.
    dphi_dt = vulkan_comp.compute_pde_rhs(y, params, grid_shape, (r, dr, dtheta))
    return dphi_dt

# -----------------
# Diagnostics Functions (unchanged from CPU version)
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
    amplitude = np.abs(phi)
    nexus_idx = np.unravel_index(np.argmin(amplitude), amplitude.shape)
    min_ampl = amplitude[nexus_idx]
    threshold = 1.5 * min_ampl
    mask = amplitude <= threshold
    core_pixels = np.argwhere(mask)
    distances = np.sqrt((core_pixels[:, 0] - nexus_idx[0])**2 + (core_pixels[:, 1] - nexus_idx[1])**2)
    core_radius = np.mean(distances) if distances.size > 0 else 0.0
    return {
        'nexus_coords': nexus_idx,
        'min_amplitude': min_ampl,
        'core_radius': core_radius
    }

def dual_resonance_features(phi, nexus_coords, window_size=20):
    row, col = nexus_coords
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, phi.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, phi.shape[1])
    zoom_region = phi[r_start:r_end, c_start:c_end]
    center = ((c_end - c_start) / 2, (r_end - r_start) / 2)
    from scipy.ndimage import map_coordinates
    radial_steps = 100
    angular_steps = 360
    r_lin = np.linspace(0, window_size, radial_steps)
    theta_lin = np.linspace(0, 2 * np.pi, angular_steps, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r_lin, theta_lin, indexing='ij')
    Xc, Yc = center
    X = Xc + r_grid * np.cos(theta_grid)
    Y = Yc + r_grid * np.sin(theta_grid)
    polar_region = map_coordinates(np.abs(zoom_region), [Y.ravel(), X.ravel()], order=1, mode='reflect')
    polar_region = polar_region.reshape((radial_steps, angular_steps))
    radial_idx = radial_steps // 2
    angular_signal = polar_region[radial_idx, :] - np.mean(polar_region[radial_idx, :])
    analytic_signal = hilbert(angular_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    circulation = phase[-1] - phase[0]
    fft_vals = np.fft.rfft(angular_signal)
    freqs = np.fft.rfftfreq(len(angular_signal), d=1)
    amplitudes = np.abs(fft_vals)
    peaks, _ = find_peaks(amplitudes, height=np.max(amplitudes) * 0.3)
    return {
        'angular_circulation': circulation,
        'fft_peaks': peaks.tolist(),
        'fft_amplitudes': amplitudes.tolist()
    }

def extended_diagnostics(t_val, y):
    phi = y.reshape((Nr, Ntheta))
    diag = {'time': t_val}
    diag.update(compute_global_statistics(phi))
    nexus_data = detect_vortex_nexus(phi)
    diag.update(nexus_data)
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
    
    # Initial field: nearly uniform with small noise.
    phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta)
    y0 = phi0.ravel()
    grid_shape = (Nr, Ntheta)
    
    # Initialize the Vulkan compute module.
    vulkan_comp = VulkanCompute()
    vulkan_comp.initialize()  # This sets up Vulkan and loads the shader(s)
    
    t_span = (t0, FINAL_TIME)
    t_eval = np.arange(t0, FINAL_TIME + DIAG_INTERVAL, DIAG_INTERVAL)
    
    sol = solve_ivp(
        fun=lambda t, y: pde_rhs_vulkan(t, y, params, grid_shape, vulkan_comp),
        t_span=t_span,
        y0=y0,
        method="Radau",
        t_eval=t_eval
    )
    
    if not sol.success:
        print("Solver failed!")
        return
    
    # Save snapshots.
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
    
    # Clean up Vulkan resources.
    vulkan_comp.cleanup()

if __name__ == "__main__":
    main()
