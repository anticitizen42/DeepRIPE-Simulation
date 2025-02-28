#!/usr/bin/env python3
"""
analyze_vortex_phase.py

This script loads the final DV‑RIPE mass–energy field (expected to be a complex field)
and visualizes the vortex itself. It plots both the amplitude and phase of the field.
A vortex is revealed by a phase singularity—a point where the phase winds by 2π.
If the field is not complex, a small imaginary part is added so that phase can be computed.

It also uses a simple heuristic (the Laplacian of the phase) to estimate the vortex core
location and marks it on the plots.

Usage:
    python analyze_vortex_phase.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

def load_field(filename="final_massenergy_field_persistent.npy"):
    """Load the simulated field from a NumPy file."""
    try:
        field = np.load(filename)
        print(f"Loaded field from {filename} with shape {field.shape}")
        return field
    except Exception as e:
        print("Error loading field:", e)
        raise

def ensure_complex(field):
    """Ensure the field is complex. If not, add a small imaginary part."""
    if not np.iscomplexobj(field):
        print("Field is not complex. Adding a small imaginary part for phase analysis.")
        field = field.astype(np.complex64)
        field += 1e-6 * 1j
    return field

def analyze_field(field):
    """
    Compute the amplitude and phase of the field.
    """
    amplitude = np.abs(field)
    phase = np.angle(field)
    return amplitude, phase

def find_vortex_core(phase):
    """
    Use the Laplacian of the phase to heuristically locate the vortex core.
    The idea is that a phase singularity will produce a sharp change.
    """
    phase_lap = laplace(phase)
    idx = np.argmax(np.abs(phase_lap))
    core = np.unravel_index(idx, phase.shape)
    return core, phase_lap

def plot_vortex(amplitude, phase, vortex_core):
    """
    Plot the field amplitude and phase side-by-side, marking the vortex core.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    im0 = axs[0].imshow(amplitude, origin="lower", cmap="viridis")
    axs[0].set_title("Field Amplitude")
    axs[0].set_xlabel("Angular Index")
    axs[0].set_ylabel("Radial Index")
    fig.colorbar(im0, ax=axs[0])
    
    im1 = axs[1].imshow(phase, origin="lower", cmap="twilight")
    axs[1].set_title("Field Phase")
    axs[1].set_xlabel("Angular Index")
    axs[1].set_ylabel("Radial Index")
    fig.colorbar(im1, ax=axs[1])
    
    # Mark the vortex core on both plots.
    for ax in axs:
        ax.plot(vortex_core[1], vortex_core[0], 'ro', markersize=10, label="Vortex Core")
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the field.
    field = load_field()
    field = ensure_complex(field)
    
    # Analyze amplitude and phase.
    amplitude, phase = analyze_field(field)
    
    # Estimate the vortex core.
    vortex_core, phase_lap = find_vortex_core(phase)
    print(f"Estimated vortex core at (row, col): {vortex_core}")
    
    # Plot the results.
    plot_vortex(amplitude, phase, vortex_core)

if __name__ == "__main__":
    main()
