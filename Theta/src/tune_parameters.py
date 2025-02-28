#!/usr/bin/env python3
"""
tune_parameters.py

This script performs a fine-grained grid search over a selected PDE parameter 
to identify the value that minimizes the vortex core radius—an indicator 
of an electron-like vortex in the DV‑RIPE mass–energy field.

For this example, we sweep the stochastic jitter amplitude (xi) over a defined range.
For each candidate xi, a short persistent GPU simulation is run (using the persistent
Vulkan compute module), and the resulting vortex core radius is computed.

The script then plots xi versus the vortex core radius and prints the best candidate.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import our persistent GPU simulation module.
from vulkan_compute_persistent import VulkanComputePersistent

# -------------------------
# Simulation Grid and Time Settings
# -------------------------
Nr = 128
Ntheta = 256
r_max = 50.0
r = np.linspace(0, r_max, Nr)
dr = r[1] - r[0]
dtheta = (2 * np.pi) / Ntheta
grid_shape = (Nr, Ntheta)
grid_info = (r, dr, dtheta)

# Base PDE parameters
base_params = {
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
    'xi': 0.001  # we'll sweep this parameter
}

# Simulation time parameters for tuning (short simulation for diagnostics)
final_time = 0.5  # seconds
dt = 0.01
num_steps = int(final_time / dt)

# -------------------------
# Diagnostic: Vortex Core Radius
# -------------------------
def analyze_core_radius(field, threshold_fraction=0.5):
    """
    Compute a diagnostic metric from a simulation field: the estimated vortex core radius.
    It finds the nexus (maximum amplitude) and then computes the average distance
    of pixels with amplitude above threshold_fraction*max from the nexus.
    """
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, amplitude.shape)
    max_val = amplitude[nexus]
    threshold = threshold_fraction * max_val
    mask = amplitude >= threshold
    y_idx, x_idx = np.indices(amplitude.shape)
    distances = np.sqrt((y_idx - nexus[0])**2 + (x_idx - nexus[1])**2)
    if np.any(mask):
        core_radius = np.mean(distances[mask])
    else:
        core_radius = np.nan
    return core_radius, nexus

# -------------------------
# Simulation Function for a Given xi
# -------------------------
def run_simulation_for_xi(xi_value):
    """
    Run a short simulation using the persistent GPU simulation module with a given xi.
    Returns a tuple: (xi_value, core_radius, nexus)
    """
    params = base_params.copy()
    params['xi'] = xi_value

    # Create initial state (using np.float32 for consistency).
    phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta).astype(np.float32)
    state = phi0.ravel()
    state_size = state.nbytes  # Nr*Ntheta*4

    # Initialize persistent GPU simulation.
    persistent_gpu = VulkanComputePersistent(state_size)
    persistent_gpu.initialize()
    persistent_gpu.update_state_on_gpu(state)

    t = 0.0
    for step in range(1, num_steps + 1):
        # Build push constants: order must match shader expectations.
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
    return xi_value, core_radius, nexus

# -------------------------
# Main Parameter Tuning Routine
# -------------------------
def main():
    # Define a fine grid of xi values to explore.
    xi_values = np.linspace(0.0001, 0.01, 20)
    results = []

    print("Starting parameter tuning for xi...")
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation_for_xi, xi) for xi in xi_values]
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                print(f"xi = {res[0]:.5f} -> Core Radius: {res[1]:.2f} pixels; Nexus: {res[2]}")
            except Exception as e:
                print("Error in simulation:", e)

    total_time = time.time() - start_time
    print(f"Parameter tuning complete in {total_time:.2f} seconds.")

    # Sort results by xi.
    results.sort(key=lambda x: x[0])
    xi_vals = [r[0] for r in results]
    core_radii = [r[1] for r in results]

    # Identify the best candidate (smallest core radius).
    best_candidate = min(results, key=lambda x: x[1])
    print(f"\nBest candidate: xi = {best_candidate[0]:.5f} with core radius {best_candidate[1]:.2f} pixels.")

    # Save the library to a JSON file.
    import json
    library = { "xi": [{"value": float(r[0]), "core_radius": float(r[1]), "nexus": list(map(int, r[2]))} for r in results] }
    with open("tuning_library.json", "w") as f:
        json.dump(library, f, indent=2)
    print("Parameter tuning library saved to tuning_library.json.")

    # Plot the results.
    plt.figure(figsize=(8, 6))
    plt.plot(xi_vals, core_radii, 'o-', lw=2)
    plt.xlabel("Jitter Amplitude (xi)")
    plt.ylabel("Vortex Core Radius (pixels)")
    plt.title("Tuning: Vortex Core Radius vs. xi")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
