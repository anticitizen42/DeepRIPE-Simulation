#!/usr/bin/env python3
"""
analyze_massenergy_field.py

This script analyzes and visualizes the DV‑RIPE mass–energy field simulation data.
It loads the final field from 'final_massenergy_field_persistent.npy' and creates:
  - A heatmap showing the absolute amplitude of the field.
  - A contour plot to highlight key features (such as the vortex nexus).
  - A spectral plot (via FFT) revealing the energy distribution across frequencies.

Run:
    python analyze_massenergy_field.py
"""

import numpy as np
import matplotlib.pyplot as plt

def load_field(filename="final_massenergy_field_persistent.npy"):
    """Load the simulated field from a NumPy file."""
    try:
        field = np.load(filename)
        print(f"Loaded field from {filename} with shape {field.shape}")
        return field
    except Exception as e:
        print("Error loading field:", e)
        raise

def plot_heatmap(field, title="Mass–Energy Field Amplitude", cmap="viridis"):
    """Display a heatmap of the absolute field amplitude."""
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(field), origin="lower", cmap=cmap, aspect="auto")
    plt.colorbar(label="|φ|")
    plt.title(title)
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")
    plt.tight_layout()
    plt.show()

def plot_contour(field, levels=20, title="Field Contours", cmap="coolwarm"):
    """Display a contour plot of the field to reveal structural features."""
    plt.figure(figsize=(8, 6))
    cp = plt.contour(field, levels=levels, cmap=cmap)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title(title)
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")
    plt.tight_layout()
    plt.show()

def plot_fft(field, title="Spectral Energy Density"):
    """Display the log magnitude of the 2D FFT of the field."""
    fft_field = np.fft.fftshift(np.fft.fft2(field))
    magnitude = np.log(np.abs(fft_field) + 1e-12)  # avoid log(0)
    plt.figure(figsize=(8, 6))
    plt.imshow(magnitude, origin="lower", cmap="inferno")
    plt.colorbar(label="Log Magnitude")
    plt.title(title)
    plt.xlabel("Frequency (Angular)")
    plt.ylabel("Frequency (Radial)")
    plt.tight_layout()
    plt.show()

def main():
    # Load the simulation field.
    field = load_field()
    # Generate visualizations.
    plot_heatmap(field)
    plot_contour(field)
    plot_fft(field)

if __name__ == "__main__":
    main()
