#!/usr/bin/env python3
"""
live_simulation.py

This script runs the DV‑RIPE mass–energy field simulation using the persistent
Vulkan compute pipeline and displays the evolving field in real time.

The simulation uses an explicit Euler integration loop. The state is kept on
persistent GPU buffers, and every few iterations a live-updating heatmap of |φ|
is displayed using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
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

# PDE parameters
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
    'xi': 0.001
}

FINAL_TIME = 5.0   # total simulation time (seconds)
dt = 0.01          # time step
num_steps = int(FINAL_TIME / dt)

# -------------------------
# Initial State
# -------------------------
# Use np.float32 for consistency with GPU buffers.
phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta).astype(np.float32)
state = phi0.ravel()  # flatten the state into a 1D array
state_size = state.nbytes  # should equal Nr * Ntheta * 4 bytes

# -------------------------
# Persistent GPU Simulation Initialization
# -------------------------
print("Initializing persistent Vulkan compute module...")
persistent_gpu = VulkanComputePersistent(state_size)
persistent_gpu.initialize()
persistent_gpu.update_state_on_gpu(state)
print("Initial state uploaded to GPU.")

# -------------------------
# Set Up Live Plot
# -------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
# Display initial field amplitude.
im = ax.imshow(np.abs(phi0), origin="lower", cmap="viridis", vmin=0, vmax=np.max(np.abs(phi0)))
ax.set_title("Live Simulation: DV-RIPE Mass–Energy Field")
ax.set_xlabel("Angular Index")
ax.set_ylabel("Radial Index")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("|φ|")
plt.show()

# -------------------------
# Simulation Loop (Explicit Euler) with Live Update
# -------------------------
t = 0.0
start_time = time.time()

try:
    for step in range(1, num_steps + 1):
        # Build push constants in the order expected by the shader:
        # [Nr, Ntheta, dr, dtheta, D_r, D_theta, lambda_e, v_e, delta_e, alpha, eta, gamma, e_gauge, beta, kappa, xi, t]
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
        
        # Update live plot every 10 steps.
        if step % 10 == 0:
            field = state.reshape((Nr, Ntheta))
            im.set_data(np.abs(field))
            ax.set_title(f"Live Simulation: t = {t:.2f} sec, L2 Norm = {norm(state):.2e}")
            plt.pause(0.001)
except KeyboardInterrupt:
    print("Simulation interrupted by user.")

end_time = time.time()
print(f"Simulation complete in {end_time - start_time:.2f} seconds.")

# Save final field.
final_field = state.reshape((Nr, Ntheta))
np.save("final_massenergy_field_live.npy", final_field)

persistent_gpu.cleanup()
plt.ioff()
plt.show()
