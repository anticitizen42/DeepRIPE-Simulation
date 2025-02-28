#!/usr/bin/env python3
"""
simulate_electron_vortex.py

This script reads the parameter sweep library (from "parameter_sweep_library.json")
to select candidate parameters that yield an electron-like vortex (i.e. a very tight vortex core).
Using the optimal candidate (here, from the "xi" sweep), it updates the base PDE parameters and
runs an extended persistent GPU simulation. The final field is saved as "final_electron_vortex.npy"
and visualized.

Run:
    python simulate_electron_vortex.py
"""

import numpy as np
import json
import time
import matplotlib.pyplot as plt
from numpy.linalg import norm

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

# Base PDE parameters (default values)
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
    'xi': 0.001  # to be updated based on sweep results
}

# Extended simulation time settings
FINAL_TIME = 5.0   # seconds
dt = 0.01          # time step
num_steps = int(FINAL_TIME / dt)

def select_candidate_from_library(filename="parameter_sweep_library.json", parameter="xi"):
    """
    Reads the JSON library and selects the candidate value for the specified parameter
    that resulted in the smallest vortex core radius.
    """
    with open(filename, "r") as f:
        library = json.load(f)
    results = library.get(parameter, [])
    if not results:
        raise ValueError(f"No results found for parameter {parameter}")
    # Choose the candidate with the smallest core radius.
    candidate = min(results, key=lambda x: x["core_radius"])
    print(f"Selected candidate for {parameter}: value = {candidate['value']:.4f}, core_radius = {candidate['core_radius']:.2f} pixels")
    return candidate["value"]

def build_push_constants(t, params):
    """
    Pack push constants in the order expected by the shader:
      [Nr, Ntheta, dr, dtheta, D_r, D_theta, lambda_e, v_e, delta_e, alpha, eta, gamma, e_gauge, beta, kappa, xi, t]
    """
    return np.array([
        Nr, Ntheta, dr, dtheta,
        params['D_r'], params['D_theta'], params['lambda_e'], params['v_e'],
        params['delta_e'], params['alpha'], params['eta'], params['gamma'],
        params['e_gauge'], params['beta'], params['kappa'], params['xi'], t
    ], dtype=np.float32).tobytes()

def simulate_electron_vortex():
    # Read the parameter sweep library and select the best candidate for xi.
    optimal_xi = select_candidate_from_library(parameter="xi")
    base_params["xi"] = optimal_xi
    print("Using updated parameters:")
    for k, v in base_params.items():
        print(f"  {k}: {v}")
    
    # Create initial state (use np.float32 for consistency).
    phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta).astype(np.float32)
    state = phi0.ravel()
    state_size = state.nbytes  # Should equal Nr * Ntheta * 4 bytes

    # Initialize persistent GPU simulation.
    persistent_gpu = VulkanComputePersistent(state_size)
    persistent_gpu.initialize()
    persistent_gpu.update_state_on_gpu(state)
    print("Initial state uploaded to GPU.")

    # Run the simulation loop (explicit Euler).
    t = 0.0
    start_time = time.time()
    for step in range(1, num_steps + 1):
        push_constants = build_push_constants(t, base_params)
        persistent_gpu.dispatch_simulation_step(push_constants)
        dphi_dt = persistent_gpu.read_state_from_gpu()
        state = state + dt * dphi_dt
        persistent_gpu.update_state_on_gpu(state)
        t += dt
        if step % 100 == 0:
            current_norm = norm(state)
            print(f"Time: {t:.2f} sec, L2 Norm: {current_norm:.4e}")
    end_time = time.time()
    print(f"Extended simulation complete in {end_time - start_time:.2f} seconds.")

    final_field = state.reshape((Nr, Ntheta))
    np.save("final_electron_vortex.npy", final_field)
    persistent_gpu.cleanup()
    
    # Visualize the final electron-like vortex.
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(final_field), origin="lower", cmap="viridis")
    plt.title("Final Electron-Like Vortex Amplitude")
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")
    plt.colorbar(label="|φ|")
    plt.show()
    
    return final_field

if __name__ == "__main__":
    simulate_electron_vortex()
