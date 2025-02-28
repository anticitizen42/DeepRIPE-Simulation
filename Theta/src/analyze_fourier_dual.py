#!/usr/bin/env python3
"""
analyze_fourier_dual.py

This script analyzes the central vortex (the "dot") of the DV‑RIPE mass–energy field
using Fourier analysis to detect dual resonant features. It performs the following steps:
  1. Loads the final simulation field (from a .npy file).
  2. Identifies the vortex nexus as the point of maximum amplitude.
  3. Extracts a zoomed region around the nexus.
  4. Extracts an angular profile (central radial line) from the zoom region.
  5. Computes the FFT of the angular profile.
  6. Uses a peak-finding algorithm to locate significant frequency peaks.
  7. Plots the angular profile and the FFT amplitude spectrum, marking the detected peaks.
  
Run:
    python analyze_fourier_dual.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_field(filename="final_massenergy_field_persistent.npy"):
    """Load the final simulation field from a NumPy file."""
    field = np.load(filename)
    print(f"Loaded field with shape: {field.shape}")
    return field

def find_nexus(field):
    """Identify the vortex nexus as the point of maximum amplitude."""
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, field.shape)
    print(f"Found nexus at (row, col): {nexus}")
    return nexus

def extract_zoom(field, center, half_size):
    """
    Extract a square zoom region around the specified center.
    half_size: half the side-length (in pixels) of the zoomed region.
    Returns the zoomed region and its bounding indices.
    """
    row, col = center
    r_start = max(row - half_size, 0)
    r_end = min(row + half_size, field.shape[0])
    c_start = max(col - half_size, 0)
    c_end = min(col + half_size, field.shape[1])
    zoom_region = field[r_start:r_end, c_start:c_end]
    return zoom_region, (r_start, r_end, c_start, c_end)

def analyze_angular_profile(zoom_region):
    """
    Extract the angular profile from the central radial line of the zoomed region,
    compute its FFT, and then find peaks in the FFT amplitude spectrum.
    Returns:
      - angular_profile: the 1D signal from the central radial line.
      - fft_freqs: frequencies corresponding to the FFT.
      - fft_ampl: amplitude spectrum of the FFT.
      - peaks: indices of detected peaks in the FFT amplitude.
    """
    nr, ntheta = zoom_region.shape
    radial_center = nr // 2
    angular_profile = zoom_region[radial_center, :]
    
    # Compute FFT of the angular profile.
    fft_vals = np.fft.rfft(angular_profile)
    fft_freqs = np.fft.rfftfreq(ntheta, d=1)  # normalized frequency (cycles per 2π)
    fft_ampl = np.abs(fft_vals)
    
    # Find peaks in the FFT amplitude; adjust threshold as needed.
    peaks, properties = find_peaks(fft_ampl, height=0.3 * np.max(fft_ampl))
    return angular_profile, fft_freqs, fft_ampl, peaks

def plot_fourier_analysis(angular_profile, fft_freqs, fft_ampl, peaks):
    """Plot the angular profile and its FFT amplitude spectrum, marking the detected peaks."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    # Angular profile.
    axs[0].plot(angular_profile, 'b-', lw=2)
    axs[0].set_title("Angular Profile (Central Radial Line)")
    axs[0].set_xlabel("Angular Index")
    axs[0].set_ylabel("Amplitude |φ|")
    
    # FFT amplitude spectrum.
    axs[1].stem(fft_freqs, fft_ampl, basefmt=" ")
    axs[1].plot(fft_freqs[peaks], fft_ampl[peaks], "ro", label="Detected Peaks")
    axs[1].set_title("FFT Amplitude Spectrum")
    axs[1].set_xlabel("Frequency (cycles per 2π)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the simulation field.
    field = load_field()
    
    # Identify the vortex nexus.
    nexus = find_nexus(field)
    
    # Extract a zoomed region around the nexus.
    half_size = 20  # Adjust this value to control zoom level.
    zoom_region, bounds = extract_zoom(field, nexus, half_size)
    
    # Analyze the angular profile.
    angular_profile, fft_freqs, fft_ampl, peaks = analyze_angular_profile(zoom_region)
    
    # Report the detected peaks.
    print("Detected FFT peaks:")
    for p in peaks:
        print(f"  Frequency: {fft_freqs[p]:.4f} cycles/2π, Amplitude: {fft_ampl[p]:.4e}")
    
    # Plot the Fourier analysis.
    plot_fourier_analysis(angular_profile, fft_freqs, fft_ampl, peaks)

if __name__ == "__main__":
    main()
