#!/usr/bin/env python3
"""
parameter_sweep_parallel.py

This script performs parallel parameter sweeps on the DV‑RIPE mass–energy field simulation.
For each parameter of interest, a range of values is explored concurrently (using multiple processes),
and a short simulation is run using our persistent GPU simulation module.
The diagnostic metric is the estimated vortex core radius computed from the final field.
Results for each parameter sweep are stored in a library (dictionary) and saved to disk as a JSON file,
so you have a record of how each parameter affects the vortex structure.

Usage:
    python parameter_sweep_parallel.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time, json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the persistent GPU simulation module.
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
    'xi': 0.001  # Example: this is one parameter we can sweep.
}

# Simulation time parameters for the sweep (short simulation to get diagnostic)
final_time = 0.5  # seconds
dt = 0.01
num_steps = int(final_time / dt)

# -------------------------
# Helper Functions for Diagnostics
# -------------------------
def analyze_core_radius(field, threshold_fraction=0.5):
    """
    Compute a diagnostic metric from a simulation field: the estimated vortex core radius.
    This is done by:
      - Finding the nexus (maximum amplitude).
      - Thresholding the amplitude at threshold_fraction * max.
      - Computing the average distance of pixels above threshold from the nexus.
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
# Simulation Function
# -------------------------
def run_simulation(param_name, param_value):
    """
    Run the persistent GPU simulation with a modified parameter.
    param_name: the key in the PDE parameter dictionary to sweep.
    param_value: the value for this simulation.
    
    Returns a tuple: (param_value, core_radius, nexus)
    """
    # Create a copy of base_params and update the target parameter.
    params = base_params.copy()
    params[param_name] = param_value
    
    # Create the initial state (using np.float32) and flatten.
    phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta).astype(np.float32)
    state = phi0.ravel()
    state_size = state.nbytes  # should be Nr * Ntheta * 4
    
    # Initialize persistent GPU simulation.
    persistent_gpu = VulkanComputePersistent(state_size)
    persistent_gpu.initialize()
    persistent_gpu.update_state_on_gpu(state)
    
    t = 0.0
    for step in range(1, num_steps + 1):
        # Build push constants.
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
    return param_value, core_radius, nexus

# -------------------------
# Main Parameter Sweep
# -------------------------
def main():
    # Define which parameters to sweep and their ranges.
    sweep_parameters = {
        "xi": np.linspace(0.0001, 0.005, 10),
        "D_r": np.linspace(0.5, 1.5, 10),
        "lambda_e": np.linspace(0.5, 1.5, 10)
    }
    
    # Library to hold sweep results.
    parameter_library = {}
    
    # We'll sweep each parameter independently.
    total_start = time.time()
    
    for param, values in sweep_parameters.items():
        print(f"\nSweeping parameter: {param}")
        futures = []
        with ProcessPoolExecutor() as executor:
            for val in values:
                futures.append(executor.submit(run_simulation, param, val))
            param_results = []
            for future in as_completed(futures):
                try:
                    res = future.result()
                    param_results.append(res)
                    print(f"Param {res[0]:.4f} -> Core Radius: {res[1]:.2f} pixels, Nexus: {res[2]}")
                except Exception as e:
                    print("Error during simulation:", e)
        # Sort results by parameter value.
        param_results.sort(key=lambda x: x[0])
        parameter_library[param] = param_results

    total_time = time.time() - total_start
    print(f"\nParallel parameter sweep complete in {total_time:.2f} seconds.")
    
    # Save the library to a JSON file.
    # Convert results to a JSON-serializable format (convert nexus tuple to list).
    json_library = {}
    for param, res_list in parameter_library.items():
        json_library[param] = [{"value": float(val),
                                "core_radius": float(core_radius),
                                "nexus": [int(n) for n in nexus]}
                                for val, core_radius, nexus in res_list]
    
    with open("parameter_sweep_library.json", "w") as f:
        json.dump(json_library, f, indent=2)
    print("Parameter sweep library saved to parameter_sweep_library.json")
    
    # Plot results for each parameter.
    num_params = len(sweep_parameters)
    fig, axs = plt.subplots(num_params, 1, figsize=(8, 4 * num_params))
    if num_params == 1:
        axs = [axs]
    for i, (param, res_list) in enumerate(parameter_library.items()):
        x_vals = [r[0] for r in res_list]
        y_vals = [r[1] for r in res_list]
        axs[i].plot(x_vals, y_vals, 'o-', lw=2)
        axs[i].set_xlabel(param)
        axs[i].set_ylabel("Vortex Core Radius (pixels)")
        axs[i].set_title(f"Effect of {param} on Vortex Core Radius")
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
