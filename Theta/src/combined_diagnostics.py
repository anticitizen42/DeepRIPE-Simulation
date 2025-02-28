#!/usr/bin/env python3
"""
combined_diagnostics.py
Version 1.7

This unified diagnostic script performs extended diagnostics on the mass–energy field
evolution. In addition to global field statistics, polar mapping, and dual resonance
analysis, this version zooms in on the suspected vortex region and computes extra
diagnostics (spin, local vorticity, etc.). 

A new function 'wavelet_analysis' has been added to analyze the wavelet scalogram 
around the vortex center. This resolves the NameError for 'wavelet_analysis'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, zoom
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit
import pywt

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
# Data Loading
# -----------------------------
def load_extended_diagnostics(filename="data/extended_diagnostics.npy"):
    if not os.path.exists(filename):
        print(f"Extended diagnostics file '{filename}' not found.")
        return None
    diagnostics = np.load(filename, allow_pickle=True)
    return diagnostics

def load_snapshot_files(snapshots_folder="data/field_snapshots"):
    import glob
    pattern = os.path.join(snapshots_folder, "field_*.npy")
    files = glob.glob(pattern)
    snapshot_data = []
    for f in files:
        try:
            # parse time from filename
            t_str = os.path.basename(f).replace("field_", "").replace(".npy", "")
            t_val = float(t_str)
            snapshot_data.append((t_val, f))
        except:
            pass
    snapshot_data.sort(key=lambda x: x[0])
    times = [sd[0] for sd in snapshot_data]
    snapshots = [np.load(sd[1]) for sd in snapshot_data]
    return times, snapshots

def load_final_snapshot():
    final_file = "data/final_massenergy_field.npy"
    if os.path.exists(final_file):
        return np.load(final_file)
    # fallback: load the last snapshot if final file not found
    times, snaps = load_snapshot_files()
    if snaps:
        return snaps[-1]
    return None

# -----------------------------
# Vortex Diagnostics Functions
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

def compute_local_gradients(field):
    grad_y, grad_x = np.gradient(field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(grad_mag), np.max(grad_mag)

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

def compute_local_vorticity(zoom_region):
    phase_map = np.zeros_like(zoom_region)
    for i in range(zoom_region.shape[0]):
        analytic_signal = hilbert(zoom_region[i, :])
        phase_map[i, :] = np.unwrap(np.angle(analytic_signal))
    dphase_dy, dphase_dx = np.gradient(phase_map)
    vorticity = np.abs(dphase_dx - dphase_dy)
    return np.mean(vorticity)

# -----------------------------
# Polar & Dual Resonance
# -----------------------------
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

# -----------------------------
# Wavelet Analysis (New)
# -----------------------------
def wavelet_analysis(field, center, zoom_half_size=20, scales_range=(1,64)):
    """
    Extracts a zoomed region around 'center', picks the central row of that region,
    and computes a Morlet continuous wavelet transform to analyze multi-scale structure.
    Returns (coeffs, scales, signal).
    """
    # 1. Zoom around center
    zoom_region = extract_zoom(field, center, window_size=zoom_half_size)
    
    # 2. Pick the central row
    row_mid = zoom_region.shape[0] // 2
    signal = zoom_region[row_mid, :]
    
    # 3. Wavelet transform
    scales = np.arange(*scales_range)
    coeffs, _ = pywt.cwt(signal, scales, 'morl')
    return coeffs, scales, signal

# -----------------------------
# Diagnostics Functions
# -----------------------------
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

def analyze_dual_modes(phi):
    # Dummy approach: do a rough "angular_circulation" measure + FFT-based detection
    # Return a dict
    # We'll pick the center row for demonstration
    center_row = phi[phi.shape[0]//2, :]
    analytic_signal = hilbert(center_row - np.mean(center_row))
    phase = np.unwrap(np.angle(analytic_signal))
    circulation = phase[-1] - phase[0]
    
    fft_vals = np.fft.rfft(center_row - np.mean(center_row))
    amps = np.abs(fft_vals)
    peaks, _ = find_peaks(amps, height=0.05)
    
    return {
        'angular_circulation': circulation,
        'fft_peaks': list(peaks),
        'fft_amplitudes': list(amps)
    }

# -----------------------------
# Main
# -----------------------------
def main():
    # 1. Load Extended Diagnostics
    diag_file = "data/extended_diagnostics.npy"
    diagnostics = load_extended_diagnostics(diag_file)
    if diagnostics is not None:
        print("Loaded Extended Diagnostics:")
        for entry in diagnostics:
            print(f"Time: {entry.get('time', 'N/A')}")
            for key, val in entry.items():
                if key != 'time':
                    print(f"  {key}: {val}")
            print("-" * 40)
    else:
        print(f"No extended diagnostics found at {diag_file}.")

    # 2. Snapshot analysis
    times, snaps = load_snapshot_files("data/field_snapshots")
    if times:
        print("\nSnapshot Times:")
        for t in times:
            print(f"  {t:.4f}")
        means = [np.mean(s) for s in snaps]
        stds = [np.std(s) for s in snaps]
        print("\nSnapshot Statistics (Mean, Std):")
        for t, m, s in zip(times, means, stds):
            print(f"Time: {t:.4f}  Mean: {m:.4f}  Std: {s:.4f}")
    else:
        print("No snapshots found.")

    # 3. Final Field & Vortex Zoom
    final_field = load_final_snapshot()
    if final_field is None:
        print("No final snapshot available.")
        return

    # Vortex center
    vortex_center = find_dot_center(final_field, mode='max')
    print(f"\nDetected vortex center at (row, col): {vortex_center}")

    zoom_region = extract_zoom(final_field, vortex_center, window_size=20)
    mean_zoom = np.mean(zoom_region)
    std_zoom = np.std(zoom_region)
    mean_grad, max_grad = compute_local_gradients(zoom_region)
    print("\nZoomed Region Statistics:")
    print(f"  Mean: {mean_zoom:.4f}")
    print(f"  Std: {std_zoom:.4f}")
    print(f"  Mean Gradient: {mean_grad:.4e}")
    print(f"  Max Gradient: {max_grad:.4e}")

    # 4. Vortex Diagnostics
    spin = compute_vortex_spin(zoom_region)
    energy = compute_vortex_energy(zoom_region)
    charge = compute_vortex_charge(zoom_region)
    local_vort = compute_local_vorticity(zoom_region)
    print("\nVortex Diagnostics:")
    print(f"  Vortex Spin (approx.): {spin:.4f}")
    print(f"  Vortex Energy (avg squared value): {energy:.4e}")
    print(f"  Vortex Charge (heuristic): {charge:.4e}")
    print(f"  Mean Local Vorticity: {local_vort:.4e}")

    # 5. Dual Resonance
    polar_r, polar_theta, polar_field = cart2polar(zoom_region, radial_steps=100, angular_steps=360)
    dual_modes = analyze_dual_resonance(polar_field)
    if dual_modes is None:
        print("Dual resonance analysis did not find two dominant peaks.")
    else:
        print("Dual resonance analysis found two dominant peaks:")
        for freq, amp in dual_modes:
            print(f"  Freq = {freq:.4f}, Amp = {amp:.4e}")

    # 6. Wavelet Analysis
    try:
        coeffs, scales, central_row_wav = wavelet_analysis(final_field, vortex_center, zoom_half_size=20)
        # You can do further processing or plotting here
        # For demonstration, we'll just print shape info
        print("\nWavelet Analysis (Morlet) on Zoomed Region's Central Row:")
        print(f"  Coeffs shape = {coeffs.shape}, scales range = {scales[0]}..{scales[-1]}")
    except Exception as e:
        print(f"Wavelet analysis error: {e}")

if __name__ == "__main__":
    main()
