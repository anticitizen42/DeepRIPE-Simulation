#!/usr/bin/env python3
"""
drivers/combined_diagnostics.py
Version 1.6.1

This unified diagnostic script performs extended diagnostics on the mass–energy field
evolution. In addition to global field statistics, polar mapping, and dual resonance
analysis, this version zooms in on the suspected vortex region and computes extra
diagnostics:
  - A high-resolution phase map.
  - A local vorticity map computed as the difference of phase gradients.
  - Vortex-specific metrics: spin, energy, heuristic charge, and mean vorticity.
  
These additional diagnostics help resolve the internal structure of the vortex and determine
if its dual resonant modes indeed cancel partially to produce the observed particle properties.

Run with:
    python -m drivers.combined_diagnostics
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, zoom
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit

# -----------------------------
# Helper function: Exponential Decay
# -----------------------------
def exponential_decay(lag, tau, A):
    return A * np.exp(-lag / tau)

def compute_exponential_decay(ac_profile):
    lags = np.arange(1, len(ac_profile))
    y = ac_profile[1:]
    if np.std(y) < 1e-8:
        return np.nan
    try:
        popt, _ = curve_fit(exponential_decay, lags, y, p0=[len(y)/2, 1.0], maxfev=1000)
        tau = popt[0]
    except Exception:
        tau = np.nan
    return tau

# -----------------------------
# Global Diagnostics Functions
# -----------------------------
def load_extended_diagnostics(filename="data/extended_diagnostics.npy"):
    if not os.path.exists(filename):
        print(f"Extended diagnostics file '{filename}' not found.")
        return None
    diagnostics = np.load(filename, allow_pickle=True)
    return diagnostics

def print_diagnostics(diagnostics):
    print("Loaded Extended Diagnostics:")
    for entry in diagnostics:
        print(f"Time: {entry.get('time', 'N/A')}")
        for key, value in entry.items():
            if key != 'time':
                print(f"  {key}: {value}")
        print("-" * 40)

def plot_trend(diagnostics, key, ylabel, title):
    times = [entry.get('time', np.nan) for entry in diagnostics]
    values = [entry.get(key, np.nan) for entry in diagnostics]
    plt.figure(figsize=(8, 4))
    plt.plot(times, values, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Snapshot Handling Functions
# -----------------------------
def extract_time_from_filename(filename):
    base = os.path.basename(filename)
    try:
        time_str = base.replace("field_", "").replace(".npy", "")
        return float(time_str)
    except Exception:
        return None

def load_snapshot_files(snapshots_folder="data/field_snapshots"):
    pattern = os.path.join(snapshots_folder, "field_*.npy")
    files = glob.glob(pattern)
    snapshot_data = []
    for f in files:
        t = extract_time_from_filename(f)
        if t is not None:
            snapshot_data.append((t, f))
    snapshot_data.sort(key=lambda x: x[0])
    times = [t for t, _ in snapshot_data]
    snapshots = [np.load(f) for _, f in snapshot_data]
    return times, snapshots

def compute_snapshot_statistics(snapshots):
    means = [np.mean(s) for s in snapshots]
    stds = [np.std(s) for s in snapshots]
    return means, stds

def plot_snapshot_statistics(times, means, stds):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, means, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Mean Field")
    plt.title("Mean Field Evolution")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(times, stds, marker='o', linestyle='-', color='orange')
    plt.xlabel("Time")
    plt.ylabel("Field Std Dev")
    plt.title("Field Variability Evolution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_snapshot_grid(times, snapshots, num_cols=4):
    num_snapshots = len(snapshots)
    num_rows = int(np.ceil(num_snapshots / num_cols))
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for i, (t, snap) in enumerate(zip(times, snapshots)):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(snap, cmap='viridis', origin='lower')
        plt.title(f"t = {t:.3f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_final_snapshot():
    final_file = "data/final_massenergy_field.npy"
    if os.path.exists(final_file):
        return np.load(final_file)
    times, snapshots = load_snapshot_files()
    if snapshots:
        return snapshots[-1]
    return None

# -----------------------------
# Vortex Zoom and Diagnostics Functions
# -----------------------------
def find_dot_center(field, mode='max'):
    if mode == 'max':
        idx = np.argmax(field)
    else:
        idx = np.argmin(field)
    return np.unravel_index(idx, field.shape)

def extract_zoom(field, center, window_size=20):
    row, col = center
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, field.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, field.shape[1])
    return field[r_start:r_end, c_start:c_end]

def upscale_region(region, scale_factor=5, order=3):
    return zoom(region, zoom=scale_factor, order=order)

def compute_local_gradients(field):
    grad_y, grad_x = np.gradient(field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(grad_mag), np.max(grad_mag)

# -----------------------------
# Polar Analysis Functions
# -----------------------------
def cart2polar(field, center=None, final_radius=None, radial_steps=100, angular_steps=360):
    Ny, Nx = field.shape
    if center is None:
        center = (Nx/2, Ny/2)
    if final_radius is None:
        final_radius = np.sqrt(max(center[0], Nx-center[0])**2 + max(center[1], Ny-center[1])**2)
    r = np.linspace(0, final_radius, radial_steps)
    theta = np.linspace(0, 2*np.pi, angular_steps, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    Xc, Yc = center
    X = Xc + r_grid * np.cos(theta_grid)
    Y = Yc + r_grid * np.sin(theta_grid)
    polar_field = map_coordinates(field, [Y.ravel(), X.ravel()], order=1, mode='reflect')
    polar_field = polar_field.reshape(r_grid.shape)
    return r_grid, theta_grid, polar_field

# -----------------------------
# Vortex-Specific Diagnostics
# -----------------------------
def compute_vortex_spin(zoom_region):
    r_grid, theta_grid, polar_zoom = cart2polar(zoom_region, radial_steps=100, angular_steps=360)
    radial_idx = polar_zoom.shape[0] // 2
    angular_signal = polar_zoom[radial_idx, :] - np.mean(polar_zoom[radial_idx, :])
    analytic_signal = hilbert(angular_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    circulation = phase[-1] - phase[0]
    vortex_spin = circulation / (2 * np.pi)
    return vortex_spin

def compute_vortex_energy(zoom_region):
    return np.sum(zoom_region**2) / zoom_region.size

def compute_vortex_charge(zoom_region):
    return -np.sum(zoom_region - np.mean(zoom_region)) / zoom_region.size

def compute_vorticity_map(phase_map):
    dphase_dy, dphase_dx = np.gradient(phase_map)
    vorticity = np.abs(dphase_dx - dphase_dy)
    return vorticity

def print_vortex_diagnostics(zoom_region):
    spin = compute_vortex_spin(zoom_region)
    energy = compute_vortex_energy(zoom_region)
    charge = compute_vortex_charge(zoom_region)
    phase_map = np.zeros_like(zoom_region)
    for i in range(zoom_region.shape[0]):
        analytic_signal = hilbert(zoom_region[i, :])
        phase_map[i, :] = np.unwrap(np.angle(analytic_signal))
    vorticity = compute_vorticity_map(phase_map)
    mean_vorticity = np.mean(vorticity)
    
    print("\nVortex Diagnostics:")
    print(f"  Vortex Spin (approx.): {spin:.4f}")
    print(f"  Vortex Energy (avg squared value): {energy:.4e}")
    print(f"  Vortex Charge (heuristic): {charge:.4e}")
    print(f"  Mean Local Vorticity: {mean_vorticity:.4e}")
    return spin, energy, charge, mean_vorticity

def plot_phase_map(region):
    phase_map = np.zeros_like(region)
    for i in range(region.shape[0]):
        analytic_signal = hilbert(region[i, :])
        phase_map[i, :] = np.unwrap(np.angle(analytic_signal))
    plt.figure(figsize=(6, 6))
    plt.imshow(phase_map, cmap='twilight', origin='lower')
    plt.title("Phase Map of Region")
    plt.colorbar(label="Phase (radians)")
    plt.tight_layout()
    plt.show()
    return phase_map

# -----------------------------
# Dual Resonance Analysis
# -----------------------------
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

def plot_angular_spectrum(polar_field, radial_index=None):
    if radial_index is None:
        radial_index = polar_field.shape[0] // 2
    angular_signal = polar_field[radial_index, :] - np.mean(polar_field[radial_index, :])
    fft_vals = np.fft.rfft(angular_signal)
    freqs = np.fft.rfftfreq(len(angular_signal), d=1)
    amplitudes = np.abs(fft_vals)
    plt.figure(figsize=(6,4))
    plt.plot(freqs, amplitudes, marker='o', linestyle='-')
    plt.xlabel("Angular Frequency (cycles/sample)")
    plt.ylabel("Amplitude")
    plt.title(f"Angular Spectrum at radial index {radial_index}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return freqs, amplitudes

# -----------------------------
# Boundary Vorticity Analysis
# -----------------------------
def compute_phase_map_from_field(field):
    phase_map = np.zeros_like(field)
    for i in range(field.shape[0]):
        analytic_signal = hilbert(field[i, :])
        phase_map[i, :] = np.unwrap(np.angle(analytic_signal))
    return phase_map

def compute_boundary_vorticity(field, side="right", window_size=20):
    Ny, Nx = field.shape
    if side == "right":
        region = field[:, Nx - window_size:Nx]
    elif side == "left":
        region = field[:, 0:window_size]
    elif side == "top":
        region = field[0:window_size, :]
    elif side == "bottom":
        region = field[Ny - window_size:Ny, :]
    else:
        raise ValueError("Invalid side. Choose from 'left', 'right', 'top', 'bottom'.")
    phase_map = compute_phase_map_from_field(region)
    dphase_dy, dphase_dx = np.gradient(phase_map)
    vorticity = np.mean(np.abs(dphase_dx - dphase_dy))
    return vorticity, phase_map, region

def plot_boundary_region(region, phase_map, side="right"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(region, cmap='viridis', origin='lower')
    plt.title(f"{side.capitalize()} Boundary Region")
    plt.colorbar(label="Field Value")
    plt.subplot(1, 2, 2)
    plt.imshow(phase_map, cmap='twilight', origin='lower')
    plt.title(f"Phase Map of {side.capitalize()} Boundary")
    plt.colorbar(label="Phase (radians)")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main Routine: Combined Diagnostics
# -----------------------------
def main():
    # 1. Load Extended Diagnostics
    diag_file = "data/extended_diagnostics.npy"
    diagnostics = load_extended_diagnostics(diag_file)
    if diagnostics is not None:
        print("Loaded Extended Diagnostics:")
        print_diagnostics(diagnostics)
        plot_trend(diagnostics, 'mean_field', "Mean Field", "Mean Field vs. Time")
        plot_trend(diagnostics, 'peak_spectral_energy', "Peak Spectral Energy", "Peak Spectral Energy vs. Time")
    else:
        print(f"No extended diagnostics found at {diag_file}.")

    # 2. Snapshot Statistics
    snapshots_folder = "data/field_snapshots"
    times, snapshots = load_snapshot_files(snapshots_folder)
    if times:
        print("\nSnapshot Times:")
        for t in times:
            print(f"  {t:.4f}")
        means, stds = compute_snapshot_statistics(snapshots)
        print("\nSnapshot Statistics (Mean, Std):")
        for t, m, s in zip(times, means, stds):
            print(f"Time: {t:.4f}  Mean: {m:.4f}  Std: {s:.4f}")
        plot_snapshot_statistics(times, means, stds)
        plot_snapshot_grid(times, snapshots, num_cols=4)
    else:
        print(f"No snapshots found in {snapshots_folder}.")

    # 3. Vortex Zoom Diagnostics
    final_field = load_final_snapshot()
    if final_field is None:
        print("No final snapshot available.")
        return

    vortex_center = find_dot_center(final_field, mode='max')
    print(f"\nDetected vortex center at (row, col): {vortex_center}")
    zoom_region = extract_zoom(final_field, vortex_center, window_size=20)
    scale_factor = 5
    high_res_region = upscale_region(zoom_region, scale_factor=scale_factor, order=3)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(zoom_region, cmap='viridis', origin='lower')
    plt.title("Original Zoomed Region Around Vortex")
    plt.colorbar(label="Field Value")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(high_res_region, cmap='viridis', origin='lower')
    plt.title(f"High-Resolution Zoom (scale factor = {scale_factor})")
    plt.colorbar(label="Field Value")
    plt.tight_layout()
    plt.show()
    
    mean_zoom = np.mean(zoom_region)
    std_zoom = np.std(zoom_region)
    mean_grad, max_grad = compute_local_gradients(zoom_region)
    print("\nZoomed Region Statistics:")
    print(f"  Mean: {mean_zoom:.4f}")
    print(f"  Std: {std_zoom:.4f}")
    print(f"  Mean Gradient: {mean_grad:.4e}")
    print(f"  Max Gradient: {max_grad:.4e}")
    
    spin = compute_vortex_spin(zoom_region)
    energy = compute_vortex_energy(zoom_region)
    charge = compute_vortex_charge(zoom_region)
    spin, energy, charge, mean_vorticity = print_vortex_diagnostics(zoom_region)
    
    # 4. Dual Resonance Analysis on Zoomed Region
    r_grid, theta_grid, polar_zoom = cart2polar(zoom_region, radial_steps=100, angular_steps=360)
    radial_idx = polar_zoom.shape[0] // 2
    dual_modes = analyze_dual_resonance(polar_zoom, radial_index=radial_idx)
    if dual_modes is not None:
        print("\nDual Resonance Analysis (central radial slice):")
        for freq, amp in dual_modes:
            print(f"  Frequency: {freq:.4f} cycles/sample, Amplitude: {amp:.4e}")
        plot_angular_spectrum(polar_zoom, radial_index=radial_idx)
    else:
        print("Dual resonance analysis did not detect two dominant modes.")
    
    # 5. Global Polar Analysis on Final Snapshot
    r_grid_full, theta_grid_full, polar_field = cart2polar(final_field, radial_steps=200, angular_steps=360)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(theta_grid_full, r_grid_full, polar_field, shading='auto', cmap='viridis')
    plt.xlabel("Angle (radians)")
    plt.ylabel("Radius")
    plt.title("Field in Polar Coordinates")
    plt.colorbar(label="Field Value")
    plt.tight_layout()
    plt.show()
    
    autocorr = np.zeros((polar_field.shape[0], polar_field.shape[1]))
    tau_values = np.zeros(polar_field.shape[0])
    for i in range(polar_field.shape[0]):
        signal = polar_field[i, :] - np.mean(polar_field[i, :])
        fft_signal = np.fft.fft(signal)
        power = np.abs(fft_signal)**2
        ac_full = np.fft.ifft(power).real
        ac = ac_full / ac_full[0]
        autocorr[i, :] = ac
        tau_values[i] = compute_exponential_decay(ac)
    
    mid_index = polar_field.shape[0] // 2
    sample_r = r_grid_full[mid_index, mid_index]
    lags = np.arange(autocorr[mid_index, :].shape[0])
    plt.figure(figsize=(6, 4))
    plt.plot(lags, autocorr[mid_index, :], marker='o', linestyle='-', label="Data")
    if not np.isnan(tau_values[mid_index]):
        fitted = exponential_decay(lags, tau_values[mid_index], 1.0)
        plt.plot(lags, fitted, linestyle='--', label=f"Exp fit, tau={tau_values[mid_index]:.2f}")
    plt.xlabel("Angular Lag (samples)")
    plt.ylabel("Autocorrelation")
    plt.title(f"Angular Autocorrelation (r ~ {sample_r:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    r_values = np.linspace(0, np.max(r_grid_full), polar_field.shape[0])
    plt.figure(figsize=(6, 4))
    plt.plot(r_values, tau_values, marker='o', linestyle='-')
    plt.xlabel("Radius")
    plt.ylabel("Decay Constant Tau (angular samples)")
    plt.title("Exponential Decay Tau vs. Radius")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\nRefined angular decay constants (tau) for each radial slice:")
    for i, tau in enumerate(tau_values):
        print(f"Radius index {i}: tau = {tau}")

    # 6. Boundary Vorticity Analysis
    vorticity_index, boundary_phase_map, boundary_region = compute_boundary_vorticity(final_field, side="right", window_size=20)
    print(f"\nBoundary Vorticity Index (right side): {vorticity_index:.4e}")
    plot_boundary_region(boundary_region, boundary_phase_map, side="right")

if __name__ == "__main__":
    main()
