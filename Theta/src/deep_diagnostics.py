#!/usr/bin/env python3
"""
deep_diagnostics.py

This script performs a deep diagnostic analysis on the final DV‑RIPE mass–energy field
data (saved as "final_massenergy_field_live.npy"). It computes:
  • Global statistics: mean, std, L2 norm, gradient magnitudes, spectral energy.
  • Vortex detection: locating the vortex nexus (max amplitude) and computing core metrics.
  • Fourier analysis: FFT of a central angular profile, with detected frequency peaks.
  • Wavelet analysis: a continuous wavelet transform (using the Morlet wavelet) on the central row.
  • Phase diagnostics: if the field is complex, it computes the phase gradient and circulation
    around the vortex core.
  
Results are output as printed diagnostics and multiple plots.
  
Usage:
    python deep_diagnostics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import sobel, laplace

# -------------------------
# Data Loading
# -------------------------
def load_field(filename="final_massenergy_field_live.npy"):
    """Load the simulation field from a NumPy file."""
    try:
        field = np.load(filename)
        print(f"Loaded field with shape: {field.shape}")
        return field
    except Exception as e:
        print("Error loading field:", e)
        raise

# -------------------------
# Global Statistics
# -------------------------
def compute_global_stats(field):
    """Compute and return global statistics of the field."""
    amplitude = np.abs(field)
    mean_val = np.mean(amplitude)
    std_val = np.std(amplitude)
    l2_norm = np.linalg.norm(amplitude)
    grad_x = sobel(amplitude, axis=1)
    grad_y = sobel(amplitude, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_grad = np.max(grad_magnitude)
    fft_field = np.fft.fft2(amplitude)
    spec_energy = np.sum(np.abs(fft_field)**2)
    
    stats = {
        "mean_amplitude": mean_val,
        "std_amplitude": std_val,
        "l2_norm": l2_norm,
        "max_gradient": max_grad,
        "total_spectral_energy": spec_energy
    }
    return stats

# -------------------------
# Vortex Detection and Core Metrics
# -------------------------
def find_vortex_nexus(field):
    """
    Identify the vortex nexus by finding the maximum amplitude in the field.
    Returns the (row, col) indices.
    """
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, amplitude.shape)
    print(f"Vortex nexus detected at {nexus} with amplitude {amplitude[nexus]:.4e}")
    return nexus

def compute_core_radius(field, threshold_fraction=0.5):
    """
    Compute the vortex core radius as the average distance of pixels with amplitude 
    above threshold_fraction * max from the vortex nexus.
    Returns (core_radius, nexus).
    """
    amplitude = np.abs(field)
    nexus = find_vortex_nexus(field)
    max_val = amplitude[nexus]
    threshold = threshold_fraction * max_val
    mask = amplitude >= threshold
    y_idx, x_idx = np.indices(amplitude.shape)
    distances = np.sqrt((y_idx - nexus[0])**2 + (x_idx - nexus[1])**2)
    if np.any(mask):
        core_radius = np.mean(distances[mask])
    else:
        core_radius = np.nan
    print(f"Estimated core radius: {core_radius:.2f} pixels (threshold at {threshold_fraction*100:.1f}% of max)")
    return core_radius, nexus

# -------------------------
# Fourier Analysis
# -------------------------
def fourier_analysis(field, nexus, zoom_half_size=20):
    """
    Extract a zoom region around the vortex nexus and perform FFT on the central row.
    Returns:
      - angular_profile: 1D signal from the central row.
      - fft_freqs: frequency values.
      - fft_ampl: amplitude spectrum.
      - peaks: indices of significant FFT peaks.
    """
    row, col = nexus
    r_start = max(row - zoom_half_size, 0)
    r_end = min(row + zoom_half_size, field.shape[0])
    c_start = max(col - zoom_half_size, 0)
    c_end = min(col + zoom_half_size, field.shape[1])
    zoom_region = field[r_start:r_end, c_start:c_end]
    central_row = zoom_region[zoom_region.shape[0] // 2, :]
    fft_vals = np.fft.rfft(central_row)
    fft_freqs = np.fft.rfftfreq(central_row.size, d=1)
    fft_ampl = np.abs(fft_vals)
    peaks, _ = find_peaks(fft_ampl, height=0.3*np.max(fft_ampl))
    return central_row, fft_freqs, fft_ampl, peaks, zoom_region

def plot_fourier(central_row, fft_freqs, fft_ampl, peaks):
    """Plot the angular profile and its FFT amplitude spectrum."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].plot(central_row, 'b-', lw=2)
    axs[0].set_title("Central Row Angular Profile")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("|φ|")
    axs[1].stem(fft_freqs, fft_ampl, basefmt=" ")
    axs[1].plot(fft_freqs[peaks], fft_ampl[peaks], "ro", label="Peaks")
    axs[1].set_title("FFT Amplitude Spectrum")
    axs[1].set_xlabel("Frequency (cycles per unit)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# Wavelet Analysis
# -------------------------
def wavelet_analysis(field, nexus, zoom_half_size=20):
    """
    Extract a tight zoom region around the vortex nexus, select the central row,
    and compute its continuous wavelet transform (CWT) using the Morlet wavelet.
    Returns:
      - coeffs: 2D array of wavelet coefficients.
      - scales: 1D array of scales.
      - central_row: the extracted 1D signal.
    """
    row, col = nexus
    r_start = max(row - zoom_half_size, 0)
    r_end = min(row + zoom_half_size, field.shape[0])
    c_start = max(col - zoom_half_size, 0)
    c_end = min(col + zoom_half_size, field.shape[1])
    zoom_region = field[r_start:r_end, c_start:c_end]
    central_row = zoom_region[zoom_region.shape[0] // 2, :]
    scales = np.arange(1, 64)
    coeffs, _ = pywt.cwt(central_row, scales, 'morl')
    return coeffs, scales, central_row

def plot_wavelet(coeffs, scales, central_row):
    """Plot the wavelet scalogram of the central row."""
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log(np.abs(coeffs) + 1e-6), origin="lower", cmap="jet",
               extent=[0, len(central_row), scales[-1], scales[0]], aspect="auto")
    plt.colorbar(label="Log Magnitude")
    plt.title("Wavelet Scalogram (Morlet)")
    plt.xlabel("Profile Index")
    plt.ylabel("Scale (inverse frequency)")
    plt.tight_layout()
    plt.show()

# -------------------------
# Phase Diagnostics (if field is complex)
# -------------------------
def phase_diagnostics(field, nexus, window_size=20):
    """
    If the field is complex, compute the phase and its gradient near the vortex.
    Returns:
      - phase: 2D phase array.
      - phase_grad: 2D array of phase gradient magnitude.
      - circulation: estimated circulation around the vortex core.
    """
    if not np.iscomplexobj(field):
        print("Field is not complex; skipping phase diagnostics.")
        return None, None, None
    
    phase = np.angle(field)
    # Compute phase gradient using sobel filters.
    grad_x = np.abs(np.gradient(phase, axis=1))
    grad_y = np.abs(np.gradient(phase, axis=0))
    phase_grad = np.sqrt(grad_x**2 + grad_y**2)
    
    # Extract a window around the vortex core.
    row, col = nexus
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, field.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, field.shape[1])
    window_phase = phase[r_start:r_end, c_start:c_end]
    
    # Estimate circulation by summing phase differences along the boundary of the window.
    top_edge = window_phase[0, :]
    bottom_edge = window_phase[-1, :]
    left_edge = window_phase[:, 0]
    right_edge = window_phase[:, -1]
    circulation = (np.sum(np.diff(top_edge)) + np.sum(np.diff(bottom_edge)) +
                   np.sum(np.diff(left_edge)) + np.sum(np.diff(right_edge)))
    
    return phase, phase_grad, circulation

def plot_phase_diagnostics(phase, phase_grad, nexus):
    """Plot phase and its gradient with the vortex core marked."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(phase, origin="lower", cmap="twilight")
    axs[0].set_title("Field Phase")
    axs[0].set_xlabel("Angular Index")
    axs[0].set_ylabel("Radial Index")
    axs[0].plot(nexus[1], nexus[0], "ro", markersize=8, label="Vortex Core")
    axs[0].legend()
    fig.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(phase_grad, origin="lower", cmap="inferno")
    axs[1].set_title("Phase Gradient Magnitude")
    axs[1].set_xlabel("Angular Index")
    axs[1].set_ylabel("Radial Index")
    axs[1].plot(nexus[1], nexus[0], "ro", markersize=8, label="Vortex Core")
    axs[1].legend()
    fig.colorbar(im1, ax=axs[1])
    
    plt.tight_layout()
    plt.show()

# -------------------------
# Main Diagnostic Routine
# -------------------------
def main():
    # Load the field.
    field = load_field()
    
    # Compute global statistics.
    stats = compute_global_stats(field)
    print("Global Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.4e}")
    
    # Identify the vortex nexus and compute core radius.
    core_radius, nexus = compute_core_radius(field, threshold_fraction=0.5)
    
    # Perform Fourier analysis on the central angular profile from a zoom region.
    central_row, fft_freqs, fft_ampl, peaks, zoom_region = fourier_analysis(field, nexus, zoom_half_size=20)
    print("FFT Analysis:")
    if len(peaks) > 0:
        for p in peaks:
            print(f"  Frequency: {fft_freqs[p]:.4f} cycles/unit, Amplitude: {fft_ampl[p]:.4e}")
    else:
        print("  No significant FFT peaks detected.")
    
    # Plot Fourier analysis.
    plot_fourier(central_row, fft_freqs, fft_ampl, peaks)
    
    # Perform Wavelet analysis.
    coeffs, scales, central_row_wav = wavelet_analysis(field, nexus, zoom_half_size=20)
    plot_wavelet(coeffs, scales, central_row_wav)
    
    # If the field is complex, perform phase diagnostics.
    phase, phase_grad, circulation = phase_diagnostics(field, nexus, window_size=20)
    if phase is not None:
        print(f"Estimated phase circulation around vortex: {circulation:.4e}")
        plot_phase_diagnostics(phase, phase_grad, nexus)
    
if __name__ == "__main__":
    main()
