#!/usr/bin/env python3
"""
detailed_fourier_spectroscopy.py

This script performs a detailed Fourier spectroscopy analysis on the DV‑RIPE mass–energy field.
It:
  1. Loads a simulation snapshot (e.g. final_massenergy_field_live.npy).
  2. Identifies the vortex nexus (maximum amplitude).
  3. Extracts a zoom region around the nexus and selects the central angular profile.
  4. Applies a Hamming window and zero-padding to increase frequency resolution.
  5. Computes the FFT and plots the amplitude and phase spectra.
  6. Detects and prints delicate frequency peaks, revealing fine-scale spectral details.

Usage:
    python detailed_fourier_spectroscopy.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_field(filename="final_massenergy_field_live.npy"):
    """Load the simulation field from a NumPy file."""
    field = np.load(filename)
    print(f"Loaded field with shape: {field.shape}")
    return field

def find_vortex_nexus(field):
    """Identify the vortex nexus as the point of maximum absolute amplitude."""
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, field.shape)
    print(f"Vortex nexus detected at: {nexus}")
    return nexus

def extract_zoom(field, center, half_size):
    """
    Extract a zoomed region around the given center.
    half_size: half the side-length of the square region.
    Returns the zoom region.
    """
    row, col = center
    r_start = max(row - half_size, 0)
    r_end = min(row + half_size, field.shape[0])
    c_start = max(col - half_size, 0)
    c_end = min(col + half_size, field.shape[1])
    zoom_region = field[r_start:r_end, c_start:c_end]
    print(f"Extracted zoom region with shape: {zoom_region.shape}")
    return zoom_region

def extract_angular_profile(zoom_region):
    """
    From a zoomed region, extract the central row (angular profile).
    """
    central_row = zoom_region[zoom_region.shape[0] // 2, :]
    return central_row

def detailed_fft(profile, pad_factor=4):
    """
    Compute a high-resolution FFT of the 1D profile.
    A Hamming window is applied, and the signal is zero-padded by pad_factor.
    Returns the frequency array, amplitude spectrum, and phase spectrum.
    """
    n = len(profile)
    window = np.hamming(n)
    profile_windowed = profile * window
    n_padded = n * pad_factor
    fft_vals = np.fft.rfft(profile_windowed, n=n_padded)
    freq = np.fft.rfftfreq(n_padded, d=1)  # assume unit spacing for now
    amplitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    return freq, amplitude, phase

def plot_fft_spectra(freq, amplitude, phase, peaks):
    """Plot the amplitude and phase spectra, marking detected peaks."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    axs[0].plot(freq, amplitude, 'b-', lw=2)
    axs[0].plot(freq[peaks], amplitude[peaks], "ro", label="Detected Peaks")
    axs[0].set_title("High-Resolution FFT Amplitude Spectrum")
    axs[0].set_xlabel("Frequency (cycles/unit)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    
    axs[1].plot(freq, phase, 'm-', lw=2)
    axs[1].set_title("FFT Phase Spectrum")
    axs[1].set_xlabel("Frequency (cycles/unit)")
    axs[1].set_ylabel("Phase (radians)")
    
    plt.tight_layout()
    plt.show()

def main():
    # Load field and identify vortex nexus.
    field = load_field()
    nexus = find_vortex_nexus(field)
    
    # Extract a zoomed region around the nexus.
    half_size = 20  # adjust as needed
    zoom_region = extract_zoom(field, nexus, half_size)
    
    # Extract the central angular profile.
    profile = extract_angular_profile(zoom_region)
    plt.figure(figsize=(8, 4))
    plt.plot(profile, 'b-', lw=2)
    plt.title("Extracted Angular Profile (Central Row of Zoom)")
    plt.xlabel("Index")
    plt.ylabel("|φ|")
    plt.tight_layout()
    plt.show()
    
    # Compute detailed FFT with windowing and zero-padding.
    freq, amplitude, phase = detailed_fft(profile, pad_factor=4)
    
    # Detect peaks in the amplitude spectrum.
    peak_threshold = 0.3 * np.max(amplitude)
    peaks, properties = find_peaks(amplitude, height=peak_threshold)
    print("Detected FFT peaks:")
    for p in peaks:
        print(f"  Frequency: {freq[p]:.4f} cycles/unit, Amplitude: {amplitude[p]:.4e}")
    
    # Plot the amplitude and phase spectra.
    plot_fft_spectra(freq, amplitude, phase, peaks)

if __name__ == "__main__":
    main()
