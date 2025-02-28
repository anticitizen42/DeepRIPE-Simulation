#!/usr/bin/env python3
"""
analyze_dual_resonance_detailed.py

This script drills into the high-energy nexus (the central peak) of the DV‑RIPE mass–energy field,
and analyzes its dual-resonant structure.

It performs the following steps:
  1. Loads a simulation snapshot (default: final_massenergy_field_persistent.npy).
  2. Identifies the nexus (by maximum absolute amplitude).
  3. Extracts a tight zoom region around the nexus.
  4. Computes an angular profile (at the radial center of the zoom).
  5. Performs FFT and Hilbert analysis on the angular profile.
  6. Visualizes the zoomed field, the angular profile, its FFT amplitude spectrum,
     and the phase profile.

Run:
    python analyze_dual_resonance_detailed.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import hilbert
import re

def load_field(filename="final_massenergy_field_persistent.npy"):
    """Load the simulated field from a NumPy file."""
    try:
        field = np.load(filename)
        print(f"Loaded field from {filename} with shape {field.shape}")
        return field
    except Exception as e:
        print("Error loading field:", e)
        raise

def find_nexus(field):
    """
    Identify the high-energy nexus.
    Returns the (row, col) indices of the maximum absolute amplitude.
    """
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, field.shape)
    print(f"Identified nexus at (row, col) = {nexus}")
    return nexus

def extract_zoom(field, center, half_size):
    """
    Extract a square zoom region around the center with half-size (in pixels).
    Returns the zoomed region and its bounding indices: (row_start, row_end, col_start, col_end).
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
    At the radial center of the zoom region, extract the angular profile,
    compute its FFT, and also compute the phase via Hilbert transform.
    Returns the angular coordinate array, profile, FFT frequencies, FFT amplitude, and phase.
    """
    nr, ntheta = zoom_region.shape
    radial_idx = nr // 2  # take the central radial line
    angular_profile = zoom_region[radial_idx, :]
    
    # Compute FFT of the angular profile.
    fft_vals = np.fft.rfft(angular_profile)
    fft_freqs = np.fft.rfftfreq(ntheta, d=1)  # normalized frequency (per pixel)
    fft_ampl = np.abs(fft_vals)
    
    # Compute analytic signal for phase.
    analytic_signal = hilbert(angular_profile)
    phase = np.unwrap(np.angle(analytic_signal))
    
    angular_coords = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    return angular_coords, angular_profile, fft_freqs, fft_ampl, phase

def plot_dual_resonance(zoom_region, angular_coords, profile, fft_freqs, fft_ampl, phase, bounds):
    """
    Create a figure with multiple panels:
      - Zoomed-in heatmap of the field around the nexus.
      - Angular profile at the center.
      - FFT amplitude spectrum of the angular profile.
      - Phase profile from the Hilbert transform.
    """
    r_start, r_end, c_start, c_end = bounds
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Zoomed-in heatmap with nexus marked.
    axs[0, 0].imshow(np.abs(zoom_region), origin="lower", cmap="viridis")
    axs[0, 0].set_title("Zoomed-In Field Amplitude")
    axs[0, 0].set_xlabel("Angular Index")
    axs[0, 0].set_ylabel("Radial Index")
    center_marker = (zoom_region.shape[1] // 2, zoom_region.shape[0] // 2)
    axs[0, 0].plot(center_marker[0], center_marker[1], 'ro', markersize=8)
    
    # Panel 2: Angular profile.
    axs[0, 1].plot(angular_coords, np.abs(profile), 'b-', lw=2)
    axs[0, 1].set_title("Angular Profile at Radial Center")
    axs[0, 1].set_xlabel("Angle (radians)")
    axs[0, 1].set_ylabel("|φ|")
    
    # Panel 3: FFT amplitude spectrum.
    axs[1, 0].stem(fft_freqs, fft_ampl, basefmt=" ")  # removed use_line_collection argument
    axs[1, 0].set_title("FFT Amplitude Spectrum of Angular Profile")
    axs[1, 0].set_xlabel("Frequency (cycles per 2π)")
    axs[1, 0].set_ylabel("Amplitude")
    
    # Panel 4: Phase profile.
    axs[1, 1].plot(angular_coords, phase, 'm-', lw=2)
    axs[1, 1].set_title("Phase Profile of Angular Signal")
    axs[1, 1].set_xlabel("Angle (radians)")
    axs[1, 1].set_ylabel("Phase (radians)")
    
    plt.tight_layout()
    plt.show()

    print(f"Global field size: {zoom_region.shape}")
    print(f"Zoom-out region bounds: {bounds} (size: {(bounds[1]-bounds[0], bounds[3]-bounds[2])})")

def main():
    # Load the field (choose a snapshot; here we use the final field).
    field = np.load("final_massenergy_field_persistent.npy")
    print(f"Field shape: {field.shape}")
    
    # Identify the nexus (peak).
    nexus = find_nexus(field)
    
    # Extract a tight zoom region around the nexus.
    zoom_in_size = 20  # half-size of zoom region
    zoom_region, bounds = extract_zoom(field, nexus, zoom_in_size)
    
    # Analyze the angular profile at the radial center of the zoom region.
    angular_coords, profile, fft_freqs, fft_ampl, phase = analyze_angular_profile(zoom_region)
    
    # Plot the double zoom and the analysis.
    plot_dual_resonance(zoom_region, angular_coords, profile, fft_freqs, fft_ampl, phase, bounds)
    
    # Optionally, print out scale information.
    print(f"Zoom region bounds: {bounds} (size: {(bounds[1]-bounds[0], bounds[3]-bounds[2])} pixels)")
    
if __name__ == "__main__":
    main()
