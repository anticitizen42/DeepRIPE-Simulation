#!/usr/bin/env python3
"""
parameter_sweep.py

This script performs a parameter sweep for the DV‑RIPE mass–energy field simulation.
It varies the stochastic jitter amplitude (xi) over a defined range, runs a short simulation
for each xi using our persistent GPU simulation module, and then computes a diagnostic metric:
the estimated vortex core radius.

The diagnostic (core radius) is computed by:
  - Loading the final simulation field,
  - Identifying the vortex nexus (point of maximum amplitude),
  - Thresholding the amplitude (e.g. at 50% of maximum),
  - Computing the average distance from the nexus for pixels above threshold.

Snapshots from each simulation are not saved here (to save time), but you can easily add that
if needed. Finally, the script plots xi versus the estimated core radius.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import the persistent GPU simulation module.
from vulkan_compute_persistent import VulkanComputePersistent

def analyze_core_radius(field, threshold_fraction=0.5):
    """
    Analyze a field snapshot to compute the vortex core radius.
    - field: 2D array (np.float32 or complex; if complex, amplitude is computed)
    - threshold_fraction: pixels with amplitude >= threshold_fraction * max_val are used.
    Returns:
       core_radius: average distance (in pixels) of these pixels from the nexus.
    """
    # If field is complex, take absolute value.
    if np.iscomplexobj(field):
        amplitude = np.abs(field)
    else:
        amplitude = field

    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, field.shape)
    max_val = amplitude[nexus]
    threshold = threshold_fraction * max_val
    mask = amplitude >= threshold

    # Create coordinate arrays.
    rows, cols = field.shape
    y_idx, x_idx = np.indices((rows, cols))
    distances = np.sqrt((y_idx - nexus[0])**2 + (x_idx - nexus[1])**2)
    if np.any(mask):
        core_radius = np.mean(distances[mask])
    else:
        core_radius = np.nan
    return core_radius, nexus

def run_simulation_for_xi(xi_value, final_time=0.5, dt=0.01):
    """
    Run the persistent GPU simulation with a given jitter amplitude (xi_value).
    Uses a reduced simulation time for parameter exploration.
    Returns the final field and the estimated vortex core radius.
    """
    Nr = 128
    Ntheta = 256
    r_max = 50.0
    r = np.linspace(0, r_max, Nr)
    dr = r[1] - r[0]
    dtheta = (2 * np.pi) / Ntheta
    grid_shape = (Nr, Ntheta)
    grid_info = (r, dr, dtheta)
    
    # Base PDE parameters; xi is the one we'll sweep.
    params = {
        'D_r': 1.0,
        'D_theta': 0.8,
        'lambda_e': 1.0,
        'v_e': 1.0,
        'delta_e': 0.1,
        'alpha': 1.0,
        'eta': 0.1,
        'gamma': 0.1,
        'e_gauge': 0.05,
        'beta': 0.0005,
        'kappa': 0.5,
        'xi': xi_value
    }
    
    num_steps = int(final_time / dt)
    
    # Create the initial state (using np.float32).
    phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta).astype(np.float32)
    state = phi0.ravel()
    state_size = state.nbytes  # Should be Nr * Ntheta * 4

    # Initialize persistent GPU simulation.
    persistent_gpu = VulkanComputePersistent(state_size)
    persistent_gpu.initialize()
    persistent_gpu.update_state_on_gpu(state)
    
    t = 0.0
    for step in range(1, num_steps + 1):
        # Build push constants; order: Nr, Ntheta, dr, dtheta, D_r, D_theta, lambda_e, v_e, delta_e, alpha, eta, gamma, e_gauge, beta, kappa, xi, time.
        push_constants = np.array([
            Nr, Ntheta, dr, dtheta,
            params['D_r'], params['D_theta'], params['lambda_e'], params['v_e'],
            params['delta_e'], params['alpha'], params['eta'], params['gamma'],
            params['e_gauge'], params['beta'], params['kappa'], params['xi'], t
        ], dtype=np.float32).tobytes()
        
        persistent_gpu.dispatch_simulation_step(push_constants)
        dphi_dt = persistent_gpu.read_state_from_gpu()
        state = state + dt * dphi_dt
        persistent_gpu.update_state_on_gpu(state)
        t += dt
    
    final_field = state.reshape((Nr, Ntheta))
    core_radius, nexus = analyze_core_radius(final_field, threshold_fraction=0.5)
    persistent_gpu.cleanup()
    return final_field, core_radius, nexus

def main():
    # Define a range of xi values to explore.
    xi_values = np.linspace(0.0001, 0.005, 10)
    core_radii = []
    nexus_list = []
    simulation_times = []
    
    start_time = time.time()
    for xi in xi_values:
        print(f"Running simulation for xi = {xi:.4f}")
        field, core_radius, nexus = run_simulation_for_xi(xi, final_time=0.5, dt=0.01)
        print(f"xi = {xi:.4f}: Vortex core radius = {core_radius:.2f} pixels, nexus at {nexus}")
        core_radii.append(core_radius)
        nexus_list.append(nexus)
        simulation_times.append(0.5)  # or record actual simulation time if desired
    
    total_time = time.time() - start_time
    print(f"Parameter sweep complete in {total_time:.2f} seconds.")
    
    # Plot the effect of xi on the vortex core radius.
    plt.figure(figsize=(8, 6))
    plt.plot(xi_values, core_radii, 'o-', lw=2)
    plt.xlabel("Jitter amplitude (xi)")
    plt.ylabel("Estimated Vortex Core Radius (pixels)")
    plt.title("Parameter Sweep: Effect of xi on Vortex Core Radius")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
