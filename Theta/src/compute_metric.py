#!/usr/bin/env python3
"""
compute_metric.py (Multi-Radius Flux Version)

Loads 'short_final.npy', detects the vortex nexus, and measures energy flux
at multiple radii around that nexus. Outputs a single float metric to stdout.

We reference real PDE logic in dvripe_physics.py:
 - compute_energy_flux(phi) -> (Sx, Sy)
 - detect_vortex_center(phi) -> (row, col)

Currently, the metric is defined as the difference between flux at the smallest
and largest radius. If you prefer a different approach (e.g., sum, average, etc.),
modify how 'metric_val' is computed.

Usage:
    python compute_metric.py
"""

import os
import sys
import numpy as np

# Import your real PDE flux & vortex detection:
# (Make sure dvripe_physics.py has these implemented, no toy code.)
from dvripe_physics import compute_energy_flux, detect_vortex_center

def ring_energy_flux(phi: np.ndarray,
                     center: tuple[int,int],
                     radius: float,
                     num_points: int=200) -> float:
    """
    Compute net flux across a ring of 'radius' around 'center' via line integral.

    Parameters
    ----------
    phi : np.ndarray
        2D array of the DV-RIPE mass-energy field.
    center : tuple[int, int]
        (row, col) of the vortex nexus.
    radius : float
        The ring radius in grid cells.
    num_points : int
        Number of samples around the ring.

    Returns
    -------
    flux_sum : float
        Net flux (S · n ds) across that ring.
        Sign convention depends on PDE flux and ring normal direction.
    """
    Sx, Sy = compute_energy_flux(phi)  # Real PDE flux from dvripe_physics

    row_c, col_c = center
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    flux_sum = 0.0

    for theta in angles:
        # parametric ring coords in array indexing
        rrow_f = row_c + radius * np.sin(theta)
        rcol_f = col_c + radius * np.cos(theta)
        rrow_i = int(round(rrow_f))
        rcol_i = int(round(rcol_f))

        # skip if out of bounds
        if (rrow_i < 0 or rrow_i >= phi.shape[0] or
            rcol_i < 0 or rcol_i >= phi.shape[1]):
            continue

        # radial unit vector in array coords: (nx, ny)
        nx = np.cos(theta)  # x direction is columns
        ny = np.sin(theta)  # y direction is rows

        Fx = Sx[rrow_i, rcol_i]
        Fy = Sy[rrow_i, rcol_i]

        dot_flux = Fx * nx + Fy * ny
        flux_sum += dot_flux

    # multiply by ring arc length segment
    delta_theta = 2.0 * np.pi / num_points
    flux_sum *= (radius * delta_theta)
    return flux_sum

def main():
    if not os.path.exists("short_final.npy"):
        # If the PDE result is missing, print a large or sentinel value
        print("999999.0")
        sys.exit(0)

    # Load final field from short_simulation
    phi = np.load("short_final.npy")

    # Detect nexus (pick your method: 'grad_min', 'local_extreme', etc.)
    center = detect_vortex_center(phi, method="grad_min")

    # Example: measure flux at multiple radii
    radius_list = [2.0, 4.0, 6.0, 8.0, 10.0]
    flux_values = []

    for r in radius_list:
        flux_r = ring_energy_flux(phi, center, r, num_points=200)
        flux_values.append(flux_r)

    # We produce a single float for the scanner.
    # Here, we do difference between flux at smallest & largest radius:
    if len(flux_values) >= 2:
        metric_val = flux_values[-1] - flux_values[0]
    else:
        metric_val = flux_values[0] if flux_values else 999999.0

    # Print the metric
    print(f"{metric_val:.6f}")

if __name__ == "__main__":
    main()
