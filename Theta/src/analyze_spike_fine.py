#!/usr/bin/env python3
"""
analyze_spike_fine.py

This script performs a fine-grained analysis of the vortex nexus (the spike) in the 
DV‑RIPE mass–energy field. It:
  1. Loads the final field.
  2. Identifies the vortex nexus (maximum amplitude).
  3. Extracts a tight window around the nexus.
  4. Selects a 1D profile (the central row) from the window.
  5. Computes a continuous wavelet transform (CWT) using a Morlet wavelet.
  6. Plots the scalogram (wavelet coefficients) to reveal multi-scale frequency details.

Run:
    python analyze_spike_fine.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt  # PyWavelets for continuous wavelet transform

def load_field(filename="final_massenergy_field_persistent.npy"):
    """Load the simulation field from a NumPy file."""
    field = np.load(filename)
    print(f"Loaded field with shape: {field.shape}")
    return field

def find_nexus(field):
    """Identify the vortex nexus as the point of maximum amplitude."""
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, amplitude.shape)
    print(f"Vortex nexus found at: {nexus}")
    return nexus

def extract_window(field, center, half_size):
    """
    Extract a square window around the specified center.
    half_size: half the side-length (in pixels) of the window.
    Returns the window and its bounding indices.
    """
    row, col = center
    r_start = max(row - half_size, 0)
    r_end = min(row + half_size, field.shape[0])
    c_start = max(col - half_size, 0)
    c_end = min(col + half_size, field.shape[1])
    window = field[r_start:r_end, c_start:c_end]
    print(f"Extracted window shape: {window.shape}")
    return window, (r_start, r_end, c_start, c_end)

def analyze_wavelet(profile):
    """
    Compute the continuous wavelet transform (CWT) of a 1D signal (profile)
    using the Morlet wavelet. Returns the wavelet coefficients and corresponding scales.
    """
    scales = np.arange(1, 64)  # adjust as needed for resolution
    coefficients, frequencies = pywt.cwt(profile, scales, 'morl')
    return coefficients, frequencies

def plot_scalogram(profile, coefficients, frequencies):
    """Plot the wavelet scalogram of the profile."""
    plt.figure(figsize=(10, 6))
    extent = [0, len(profile), frequencies[-1], frequencies[0]]
    plt.imshow(np.abs(coefficients), extent=extent, cmap='jet', aspect='auto', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Scalogram (Morlet) of Central Profile')
    plt.xlabel('Profile Index')
    plt.ylabel('Scale (inverse frequency)')
    plt.tight_layout()
    plt.show()

def main():
    # Load the simulation field.
    field = load_field()
    
    # Identify the vortex nexus.
    nexus = find_nexus(field)
    
    # Extract a tight window around the nexus (e.g., half-size 10 pixels).
    window, bounds = extract_window(field, nexus, half_size=10)
    
    # For fine-grained analysis, we focus on the central row of the window.
    central_row = window[window.shape[0] // 2, :]
    
    # Plot the extracted 1D profile.
    plt.figure(figsize=(8, 4))
    plt.plot(central_row, 'b-', lw=2)
    plt.title("Central Row of Extracted Window")
    plt.xlabel("Index")
    plt.ylabel("|φ|")
    plt.tight_layout()
    plt.show()
    
    # Compute the continuous wavelet transform.
    coeffs, freqs = analyze_wavelet(central_row)
    
    # Plot the scalogram.
    plot_scalogram(central_row, coeffs, freqs)

if __name__ == "__main__":
    main()
