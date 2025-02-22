#!/usr/bin/env python3
"""
sim_diag_winding.py

This script diagnoses the vortex structure in the electron field by:
  1. Initializing the electron fields using the updated fields.py.
  2. Extracting the central 2D slice (assumed to host the vortex) from phi1.
  3. Computing the topological winding number by summing phase differences along the perimeter.
  4. Writing detailed output to the console for inspection.

Run this script to verify that your vortex seeding produces the expected phase winding.
"""

import numpy as np
from src.fields import init_electron_fields

def compute_winding(phi_slice):
    """
    Compute the winding number (topological charge) of a 2D complex field phi_slice.
    The winding is determined by summing the phase differences along the boundary of phi_slice.
    
    Parameters:
      phi_slice: 2D numpy array (complex) representing a slice of the electron field.
    
    Returns:
      winding_number: float, the computed winding number.
    """
    phase = np.angle(phi_slice)
    Ny, Nz = phi_slice.shape
    winding = 0.0

    # Top edge (left to right)
    for j in range(0, Nz - 1):
        diff = np.angle(np.exp(1j * (phase[0, j+1] - phase[0, j])))
        winding += diff

    # Right edge (top to bottom)
    for i in range(0, Ny - 1):
        diff = np.angle(np.exp(1j * (phase[i+1, Nz-1] - phase[i, Nz-1])))
        winding += diff

    # Bottom edge (right to left)
    for j in range(Nz-1, 0, -1):
        diff = np.angle(np.exp(1j * (phase[Ny-1, j-1] - phase[Ny-1, j])))
        winding += diff

    # Left edge (bottom to top)
    for i in range(Ny-1, 0, -1):
        diff = np.angle(np.exp(1j * (phase[i-1, 0] - phase[i, 0])))
        winding += diff

    # Normalize by 2*pi to get the winding number
    winding_number = winding / (2 * np.pi)
    return winding_number

def dump_phase_stats(phi_slice):
    """
    Compute and return basic statistics for the phase of the field.
    """
    phase = np.angle(phi_slice)
    return {
        "min": np.min(phase),
        "max": np.max(phase),
        "mean": np.mean(phase),
        "std": np.std(phase)
    }

if __name__ == "__main__":
    # Use the same field shape as in your simulation
    field_shape = (4, 8, 16, 16)  # (N0, N1, Ny, Nz)
    phi1, phi2 = init_electron_fields(field_shape)
    
    # Extract the central slice along the first two dimensions
    central_phi1 = phi1[field_shape[0] // 2, field_shape[1] // 2, :, :]

    # Dump amplitude and phase statistics for the central slice
    amplitude = np.abs(central_phi1)
    phase_stats = dump_phase_stats(central_phi1)
    print("Central slice amplitude statistics:")
    print("  min: {:.4e}, max: {:.4e}, mean: {:.4e}, std: {:.4e}".format(
        np.min(amplitude), np.max(amplitude), np.mean(amplitude), np.std(amplitude)))
    print("Central slice phase statistics:")
    print("  min: {:.4e}, max: {:.4e}, mean: {:.4e}, std: {:.4e}".format(
        phase_stats["min"], phase_stats["max"], phase_stats["mean"], phase_stats["std"]))

    # Compute the winding number
    winding = compute_winding(central_phi1)
    print("\nComputed winding number (topological charge / spin) from central slice of phi1: {:.4f}".format(winding))
