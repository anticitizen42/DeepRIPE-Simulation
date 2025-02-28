#!/usr/bin/env python3
"""
live_simulation_wavelet_animated.py

This script runs the DV‑RIPE mass–energy field simulation using our persistent Vulkan
compute pipeline with gravitational and membrane coupling terms. It displays two live-updating plots:
  1. A heatmap of the field amplitude |φ|.
  2. A continuously updated wavelet scalogram of the central row (using a Morlet wavelet).

The simulation runs until you interrupt it with Ctrl+C. Final simulation data is saved for later analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
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

# PDE parameters (base values)
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

dt = 0.01  # time step
diagnostic_interval = 10  # update plots every 10 steps

# -------------------------
# Initial State
# -------------------------
phi0 = np.ones((Nr, Ntheta), dtype=np.float32) + 0.01 * np.random.randn(Nr, Ntheta).astype(np.float32)
state = phi0.ravel()
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
# Push Constants Builder
# -------------------------
def build_push_constants(t, state):
    """
    Build push constants for the shader in the following order:
      [Nr, Ntheta, dr, dtheta,
       D_r, D_theta, lambda_e, v_e,
       delta_e, alpha, eta, gamma,
       e_gauge, beta, kappa, xi,
       global_density, boundary_avg, t]
    global_density: mean(|φ|²) over entire field.
    boundary_avg: average of φ along the outer radial boundary.
    """
    field = state.reshape((Nr, Ntheta))
    global_density = np.mean(np.abs(field)**2)
    boundary_avg = np.mean(field[-1, :])
    return np.array([
        Nr, Ntheta, dr, dtheta,
        params['D_r'], params['D_theta'], params['lambda_e'], params['v_e'],
        params['delta_e'], params['alpha'], params['eta'], params['gamma'],
        params['e_gauge'], params['beta'], params['kappa'], params['xi'],
        global_density, boundary_avg, t
    ], dtype=np.float32).tobytes()

# -------------------------
# Wavelet Analysis Function
# -------------------------
def compute_wavelet_scalogram(state):
    """
    Extract the central row from the field and compute its continuous wavelet transform (CWT)
    using the Morlet wavelet.
    Returns:
      coefficients: 2D array of wavelet coefficients,
      scales: 1D array of scales,
      central_row: the extracted 1D profile.
    """
    field = state.reshape((Nr, Ntheta))
    central_row = field[Nr // 2, :]
    scales = np.arange(1, 64)  # adjust scales as needed
    coeffs, _ = pywt.cwt(central_row, scales, 'morl')
    return coeffs, scales, central_row

# -------------------------
# Set Up Live Plots
# -------------------------
plt.ion()
# Figure for the field heatmap.
fig1, ax1 = plt.subplots(figsize=(8, 6))
field_im = ax1.imshow(np.abs(phi0), origin="lower", cmap="viridis", vmin=0, vmax=np.max(np.abs(phi0)))
ax1.set_title("Live Simulation: DV-RIPE Field Amplitude")
ax1.set_xlabel("Angular Index")
ax1.set_ylabel("Radial Index")
cbar1 = fig1.colorbar(field_im, ax=ax1)
cbar1.set_label("|φ|")

# Figure for the wavelet scalogram.
fig2, ax2 = plt.subplots(figsize=(8, 6))
scalogram_im = None
ax2.set_title("Wavelet Scalogram of Central Row (Morlet)")
ax2.set_xlabel("Profile Index")
ax2.set_ylabel("Scale (inverse frequency)")
plt.show()

# -------------------------
# Simulation Loop (runs until Ctrl+C)
# -------------------------
t = 0.0
step = 0
start_time = time.time()

try:
    while True:
        push_constants = build_push_constants(t, state)
        persistent_gpu.dispatch_simulation_step(push_constants)
        dphi_dt = persistent_gpu.read_state_from_gpu()
        state = state + dt * dphi_dt
        persistent_gpu.update_state_on_gpu(state)
        t += dt
        step += 1
        
        if step % diagnostic_interval == 0:
            # Update field heatmap.
            field = state.reshape((Nr, Ntheta))
            field_im.set_data(np.abs(field))
            ax1.set_title(f"Live Simulation: t = {t:.2f} sec, L2 Norm = {norm(state):.2e}")
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            
            # Compute wavelet scalogram.
            coeffs, scales, central_row = compute_wavelet_scalogram(state)
            scalogram = np.log(np.abs(coeffs) + 1e-6)
            if scalogram_im is None:
                scalogram_im = ax2.imshow(scalogram, origin="lower", cmap="jet",
                                          extent=[0, len(central_row), scales[-1], scales[0]],
                                          aspect="auto")
                fig2.colorbar(scalogram_im, ax=ax2, label="Log Magnitude")
            else:
                scalogram_im.set_data(scalogram)
            ax2.set_title(f"Wavelet Scalogram (t = {t:.2f} sec)")
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            
            plt.pause(0.001)
except KeyboardInterrupt:
    print("Simulation interrupted by user.")

end_time = time.time()
print(f"Simulation ran for {t:.2f} seconds (wall-clock time: {end_time - start_time:.2f} sec).")

# Save final field for offline analysis.
final_field = state.reshape((Nr, Ntheta))
np.save("final_massenergy_field_live.npy", final_field)
persistent_gpu.cleanup()

plt.ioff()
plt.show()
