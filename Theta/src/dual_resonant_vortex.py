#!/usr/bin/env python3
"""
dual_resonant_vortex.py

This script analyzes the vortex (the central high-energy spike) in the DV‑RIPE mass–energy field
to detect a dual resonant structure. It:
  1. Loads a simulation field from a NumPy file (e.g. final_massenergy_field_live.npy).
  2. Identifies the vortex nexus (point of maximum amplitude).
  3. Extracts a zoomed region around the nexus and obtains the central angular profile.
  4. Computes a high-resolution FFT of the angular profile using windowing and zero-padding.
  5. Detects and annotates two significant peaks in the FFT amplitude spectrum.
  6. Plots the angular profile and FFT spectrum with annotations.

Run:
    python dual_resonant_vortex.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
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
    nexus = np.unravel_index(idx, amplitude.shape)
    print(f"Vortex nexus detected at: {nexus} with amplitude {amplitude[nexus]:.4e}")
    return nexus

def extract_zoom(field, center, half_size):
    """
    Extract a zoomed square region around the center.
    half_size: half the side-length of the region in pixels.
    Returns the zoom region.
    """
    row, col = center
    r_start = max(row - half_size, 0)
    r_end = min(row + half_size, field.shape[0])
    c_start = max(col - half_size, 0)
    c_end = min(col + half_size, field.shape[1])
    zoom_region = field[r_start:r_end, c_start:c_end]
    print(f"Extracted zoom region of shape: {zoom_region.shape}")
    return zoom_region

def extract_angular_profile(zoom_region):
    """Extract the central row (angular profile) from the zoomed region."""
    central_row = zoom_region[zoom_region.shape[0] // 2, :]
    return central_row

def detailed_fft(profile, pad_factor=4):
    """
    Compute a high-resolution FFT of a 1D profile.
    Applies a Hamming window and zero-padding by pad_factor.
    Returns frequency, amplitude, and phase arrays.
    """
    n = len(profile)
    window = np.hamming(n)
    profile_windowed = profile * window
    n_padded = n * pad_factor
    fft_vals = np.fft.rfft(profile_windowed, n=n_padded)
    freq = np.fft.rfftfreq(n_padded, d=1)  # assume unit spacing
    amplitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    return freq, amplitude, phase

def detect_dual_resonance(freq, amplitude, peak_threshold=0.3):
    """
    Detect significant peaks in the FFT amplitude spectrum.
    The threshold is given as a fraction of the maximum amplitude.
    Returns indices of peaks.
    """
    threshold = peak_threshold * np.max(amplitude)
    peaks, properties = find_peaks(amplitude, height=threshold)
    return peaks, properties

def plot_dual_resonance(profile, freq, amplitude, phase, peaks):
    """Plot the angular profile and its FFT amplitude and phase spectra, marking detected peaks."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot angular profile.
    axs[0].plot(profile, 'b-', lw=2)
    axs[0].set_title("Central Angular Profile")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("|φ|")
    
    # Plot FFT amplitude spectrum.
    axs[1].plot(freq, amplitude, 'b-', lw=2, label="FFT Amplitude")
    if len(peaks) >= 2:
        # Highlight the two highest peaks.
        peak_amplitudes = amplitude[peaks]
        top_two_indices = peaks[np.argsort(peak_amplitudes)[-2:]]
        axs[1].plot(freq[top_two_indices], amplitude[top_two_indices], "ro", markersize=8, label="Top 2 Peaks")
    elif len(peaks) > 0:
        axs[1].plot(freq[peaks], amplitude[peaks], "ro", markersize=8, label="Detected Peaks")
    axs[1].set_title("High-Resolution FFT Amplitude Spectrum")
    axs[1].set_xlabel("Frequency (cycles/unit)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    
    # Plot FFT phase spectrum.
    axs[2].plot(freq, phase, 'm-', lw=2)
    axs[2].set_title("FFT Phase Spectrum")
    axs[2].set_xlabel("Frequency (cycles/unit)")
    axs[2].set_ylabel("Phase (radians)")
    
    plt.tight_layout()
    plt.show()

def main():
    field = load_field()
    nexus = find_vortex_nexus(field)
    zoom_region = extract_zoom(field, nexus, half_size=20)
    profile = extract_angular_profile(zoom_region)
    
    # Plot the extracted profile.
    plt.figure(figsize=(8, 4))
    plt.plot(profile, 'b-', lw=2)
    plt.title("Extracted Angular Profile")
    plt.xlabel("Index")
    plt.ylabel("|φ|")
    plt.tight_layout()
    plt.show()
    
    freq, amplitude, phase = detailed_fft(profile, pad_factor=4)
    peaks, properties = detect_dual_resonance(freq, amplitude, peak_threshold=0.3)
    
    if len(peaks) == 0:
        print("No significant FFT peaks detected.")
    else:
        print("Detected FFT peaks:")
        for p in peaks:
            print(f"  Frequency: {freq[p]:.4f} cycles/unit, Amplitude: {amplitude[p]:.4e}")
    
    plot_dual_resonance(profile, freq, amplitude, phase, peaks)

if __name__ == "__main__":
    main()
