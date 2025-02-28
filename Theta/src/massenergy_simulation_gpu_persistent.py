#!/usr/bin/env python3
"""
massenergy_simulation_gpu_persistent.py
Updated to reference real PDE logic from dvripe_physics (no toy PDE code).
Runs indefinitely until you close the window or press Ctrl+C.

Features:
  - Persistent Vulkan compute for PDE updates.
  - PDE parameters are loaded from a dict or partial config.
  - The PDE field is initialized via dvripe_physics.initialize_field().
  - Each step calls your PDE solver function (or dispatch to the GPU).
  - Live plots of field amplitude and wavelet scalogram.

Ensure dvripe_physics.py has the real PDE logic:
 - initialize_field(params, Nr, Ntheta) -> np.ndarray
 - run_pde_step_gpu(...) or dispatch to your GPU kernel

Usage:
    python massenergy_simulation_gpu_persistent.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm

# Real PDE logic references:
from dvripe_physics import initialize_field  # GPU PDE or partial CPU init

# Import your persistent GPU compute module
from vulkan_compute_persistent import VulkanComputePersistent
import pywt  # wavelet for scalogram

# -------------------------
# Grid & Domain Settings
# -------------------------
Nr = 128
Ntheta = 256
r_max = 50.0
r = np.linspace(0, r_max, Nr)
dtheta = (2*np.pi) / Ntheta
dr = r[1] - r[0]
grid_shape = (Nr, Ntheta)

# PDE Parameter Dictionary (Adjust as needed)
params = {
    'D_r': 1.0,
    'D_theta': 0.8,
    'lambda_e': 1.0,
    'v_e': 1.0,
    'delta_e': 0.2,
    'alpha': 1.0,
    'eta': 0.2,
    'gamma': 0.15,
    'e_gauge': 0.08,
    'beta': 0.0005,
    'kappa': 0.5,
    'xi': 0.001
}

# -------------------------
# Live Plot Setup
# -------------------------
plt.ion()
fig_field, ax_field = plt.subplots(figsize=(8, 6))
field_im = None
cbar_field = None

fig_wavelet, ax_wavelet = plt.subplots(figsize=(8, 6))
scalogram_im = None

def update_field_plot(phi, t):
    global field_im, cbar_field
    if field_im is None:
        field_im = ax_field.imshow(np.abs(phi), origin="lower", cmap="viridis",
                                   vmin=0, vmax=np.max(np.abs(phi)))
        cbar_field = fig_field.colorbar(field_im, ax=ax_field)
        cbar_field.set_label("|φ|")
    else:
        field_im.set_data(np.abs(phi))
        field_im.set_clim(0, np.max(np.abs(phi)))

    ax_field.set_title(f"Live Simulation: t = {t:.2f} sec, L2 Norm = {norm(phi.ravel()):.2e}")
    fig_field.canvas.draw()
    fig_field.canvas.flush_events()

def update_wavelet_scalogram(phi, t):
    global scalogram_im
    central_row = phi[Nr//2, :]
    scales = np.arange(1, 64)
    coeffs, _ = pywt.cwt(central_row, scales, 'morl')
    scalogram = np.log(np.abs(coeffs) + 1e-6)

    if scalogram_im is None:
        scalogram_im = ax_wavelet.imshow(scalogram, origin="lower", cmap="jet",
                                         extent=[0, len(central_row), scales[-1], scales[0]],
                                         aspect="auto")
        fig_wavelet.colorbar(scalogram_im, ax=ax_wavelet, label="Log Magnitude")
    else:
        scalogram_im.set_data(scalogram)

    ax_wavelet.set_title(f"Wavelet Scalogram (t = {t:.2f} sec)")
    fig_wavelet.canvas.draw()
    fig_wavelet.canvas.flush_events()

def build_push_constants(t, state):
    """
    Example push constants. 
    Make sure these match your GPU shader layout for PDE logic.
    """
    phi = state.reshape((Nr, Ntheta))
    global_density = np.mean(np.abs(phi)**2)
    boundary_avg = np.mean(phi[-1, :])

    # order: [Nr, Ntheta, dr, dtheta,
    #         D_r, D_theta, lambda_e, v_e,
    #         delta_e, alpha, eta, gamma,
    #         e_gauge, beta, kappa, xi,
    #         global_density, boundary_avg, t]
    arr = np.array([
        Nr, Ntheta, dr, dtheta,
        params['D_r'], params['D_theta'], params['lambda_e'], params['v_e'],
        params['delta_e'], params['alpha'], params['eta'], params['gamma'],
        params['e_gauge'], params['beta'], params['kappa'], params['xi'],
        global_density, boundary_avg, t
    ], dtype=np.float32)
    return arr.tobytes()

def main():
    print("Initializing persistent Vulkan compute module...")
    # Flatten state for GPU
    Nx, Ny = Nr, Ntheta
    phi0 = initialize_field(params, Nx, Ny)  # Real PDE init
    state = phi0.ravel().astype(np.float32)
    state_size = state.nbytes

    persistent_gpu = VulkanComputePersistent(state_size)
    persistent_gpu.initialize()
    persistent_gpu.update_state_on_gpu(state)
    print("Initial state uploaded to GPU.")

    dt = 0.01
    t = 0.0
    step = 0

    print("Running simulation... (close plot or press Ctrl+C to terminate)")
    start_time = time.time()

    try:
        while True:
            # PDE step on GPU
            push_constants = build_push_constants(t, state)
            persistent_gpu.dispatch_simulation_step(push_constants)

            # read derivative or updated field
            dphi_dt = persistent_gpu.read_state_from_gpu()
            if dphi_dt.shape[0] != state.shape[0]:
                dphi_dt = dphi_dt[:state.shape[0]]

            # Euler update (or partial PDE approach)
            state = state + dt * dphi_dt
            persistent_gpu.update_state_on_gpu(state)
            t += dt
            step += 1

            # Update plots every 100 steps
            if step % 100 == 0:
                phi2d = state.reshape((Nr, Ntheta))
                update_field_plot(phi2d, t)
                update_wavelet_scalogram(phi2d, t)
                print(f"t = {t:.2f} sec")
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    except Exception as e:
        print("Error during simulation:", e)

    end_time = time.time()
    print(f"Simulation terminated at t = {t:.2f} sec (wall-clock time: {end_time - start_time:.2f} s).")

    final_field = state.reshape((Nr, Ntheta))
    np.save("final_electron_vortex.npy", final_field)
    persistent_gpu.cleanup()
    plt.ioff()
    plt.show()

    print("Final field saved as 'final_electron_vortex.npy'.")

if __name__ == "__main__":
    main()
