#!/usr/bin/env python3
"""
drivers/vortex_characteristics.py
Version 1.0

This script analyzes the vortex extracted from the final electron field snapshot.
It performs the following tasks:
  1. Loads the final electron field snapshot (assumed to be stored as a complex npy file).
  2. Locates the vortex center by finding the maximum magnitude (assumed to be a bright vortex).
  3. Extracts a zoomed subregion around that center and upscales it for high‐resolution viewing.
  4. Converts the zoom region to polar coordinates and computes the phase winding along a circle
     (to estimate the spin).
  5. Computes a proxy for the charge from phase gradients.
  6. Computes an integrated energy estimate from the gradient energy density.
  
Target values are: Spin ~0.5, Charge ~–1 (normalized), Energy ~1.
Adjust the formulas and parameters as needed for your model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, map_coordinates, gaussian_gradient_magnitude

# -----------------------------
# File Loading and Basic Zoom Functions
# -----------------------------
def load_final_electron_field():
    """Load the final electron field snapshot (complex array) from a dedicated file."""
    filename = "data/final_electron_field.npy"
    if os.path.exists(filename):
        field = np.load(filename)
        return field
    else:
        print(f"File {filename} not found.")
        return None

def find_vortex_center(field, mode='max'):
    """
    Locate the vortex in the field.
    For a complex field, we use the magnitude.
    """
    mag = np.abs(field)
    if mode == 'max':
        idx = np.argmax(mag)
    else:
        idx = np.argmin(mag)
    return np.unravel_index(idx, field.shape)

def extract_zoom(field, center, window_size=20):
    """
    Extract a square subregion around the given center.
    """
    row, col = center
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, field.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, field.shape[1])
    return field[r_start:r_end, c_start:c_end]

def upscale_region(region, scale_factor=5, order=3):
    """
    Upscale the region using interpolation (spline of given order).
    """
    return zoom(region, zoom=scale_factor, order=order)

def plot_field(field, title="Field", cmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(field), cmap=cmap, origin='lower')
    plt.title(title)
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Polar Mapping and Phase Winding
# -----------------------------
def cart2polar(field, center=None, final_radius=None, radial_steps=100, angular_steps=360):
    """
    Convert a 2D Cartesian field into polar coordinates.
    For a complex field, both the real and imaginary parts are interpolated.
    """
    Ny, Nx = field.shape
    if center is None:
        center = (Nx/2, Ny/2)
    if final_radius is None:
        final_radius = np.sqrt((max(center[0], Nx-center[0]))**2 + (max(center[1], Ny-center[1]))**2)
    
    r = np.linspace(0, final_radius, radial_steps)
    theta = np.linspace(0, 2*np.pi, angular_steps, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    
    Xc, Yc = center
    X = Xc + r_grid * np.cos(theta_grid)
    Y = Yc + r_grid * np.sin(theta_grid)
    
    # Interpolate the real and imaginary parts separately.
    real_interp = map_coordinates(np.real(field), [Y.ravel(), X.ravel()], order=1, mode='reflect')
    imag_interp = map_coordinates(np.imag(field), [Y.ravel(), X.ravel()], order=1, mode='reflect')
    polar_field = real_interp.reshape(r_grid.shape) + 1j * imag_interp.reshape(r_grid.shape)
    
    return r_grid, theta_grid, polar_field

def compute_phase_winding(polar_field):
    """
    Compute the phase winding (in turns) for each radial slice.
    The phase winding is calculated by unwrapping the phase along the angular axis and
    computing the net change in phase divided by 2π.
    """
    phases = np.angle(polar_field)
    unwrapped = np.unwrap(phases, axis=1)
    # The winding for each radius is the phase difference over one full circle divided by 2π.
    winding = (unwrapped[:, -1] - unwrapped[:, 0]) / (2 * np.pi)
    return winding

# -----------------------------
# Energy and Charge Diagnostics
# -----------------------------
def compute_energy(field):
    """
    Compute a simple energy estimate.
    Here we use 0.5 * |grad|^2 as a proxy for energy density.
    """
    grad_y, grad_x = np.gradient(np.abs(field))
    energy_density = 0.5 * (grad_x**2 + grad_y**2)
    total_energy = np.sum(energy_density)
    return total_energy

def compute_charge(field):
    """
    Compute a proxy for the charge.
    For a complex field, a common proxy is to integrate the divergence of the phase gradient.
    This is only a proxy—adjust as needed for your model.
    """
    phases = np.angle(field)
    unwrapped = np.unwrap(phases)
    grad_y, grad_x = np.gradient(unwrapped)
    charge_density = grad_x + grad_y
    total_charge = np.sum(charge_density)
    return total_charge

# -----------------------------
# Main Routine
# -----------------------------
def main():
    # 1. Load the final electron field snapshot.
    field = load_final_electron_field()
    if field is None:
        return
    print("Final electron field loaded.")
    
    # 2. Locate the vortex.
    vortex_center = find_vortex_center(field, mode='max')
    print(f"Detected vortex center at (row, col): {vortex_center}")
    
    # 3. Extract a zoomed region around the vortex.
    window_size = 20  # Adjust as needed.
    zoom_field = extract_zoom(field, vortex_center, window_size=window_size)
    plot_field(zoom_field, title="Original Zoomed Vortex Region")
    
    # 4. Upscale the zoomed region.
    scale_factor = 5
    high_res_field = upscale_region(zoom_field, scale_factor=scale_factor, order=3)
    plot_field(high_res_field, title=f"High-Resolution Zoom (Scale Factor {scale_factor})")
    
    # 5. Polar mapping and phase winding analysis.
    # Convert the original zoom region to polar coordinates.
    r_grid, theta_grid, polar_field = cart2polar(zoom_field, radial_steps=100, angular_steps=360)
    # For a selected radius (here, the middle), compute the phase winding.
    winding = compute_phase_winding(polar_field)
    mid_index = polar_field.shape[0] // 2
    print(f"Phase winding at mid-radius (index {mid_index}): {winding[mid_index]:.4f} turns")
    # Assuming 2 full turns corresponds to a 0.5 spin (due to double covering), we compute:
    estimated_spin = winding[mid_index] / 2.0
    print(f"Estimated Spin: {estimated_spin:.4f} (target ~0.5)")
    
    # 6. Compute energy and charge.
    total_energy = compute_energy(zoom_field)
    total_charge = compute_charge(zoom_field)
    print(f"Estimated Energy: {total_energy:.4e} (target ~1)")
    print(f"Estimated Charge (proxy): {total_charge:.4e} (target ~-1)")
    
if __name__ == "__main__":
    main()
