#!/usr/bin/env python3
"""
drivers/zoom_highres.py
Version 1.1

This script loads the final mass–energy field snapshot (from "data/final_massenergy_field.npy"
or the latest snapshot available), automatically locates the concentrated vortex region by
finding the maximum field value, and then extracts and upscales a small subregion around that point.
It also computes local gradient statistics for the zoomed region, so you can resolve fine detail
inside the vortex.

Usage:
    python -m drivers.zoom_highres
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, map_coordinates, gaussian_gradient_magnitude

def load_final_snapshot():
    """Load the final snapshot from a dedicated file if available, otherwise from snapshots folder."""
    final_file = "data/final_massenergy_field.npy"
    if os.path.exists(final_file):
        return np.load(final_file)
    snapshots_folder = "data/field_snapshots"
    if os.path.exists(snapshots_folder):
        files = sorted([f for f in os.listdir(snapshots_folder) if f.startswith("field_") and f.endswith(".npy")])
        if files:
            latest = os.path.join(snapshots_folder, files[-1])
            return np.load(latest)
    return None

def find_dot_center(field, mode='max'):
    """
    Locate the concentrated dot in the field.
    
    Parameters:
      field (np.ndarray): 2D array.
      mode (str): 'max' for a bright vortex (maximum), 'min' for a dark vortex.
      
    Returns:
      tuple: (row, col) of the detected vortex center.
    """
    if mode == 'max':
        idx = np.argmax(field)
    else:
        idx = np.argmin(field)
    return np.unravel_index(idx, field.shape)

def extract_zoom(field, center, window_size=20):
    """
    Extract a subregion of the field around the given center.
    
    Parameters:
      field (np.ndarray): 2D field.
      center (tuple): (row, col) coordinates.
      window_size (int): Half-width of the window.
      
    Returns:
      np.ndarray: Extracted zoomed region.
    """
    row, col = center
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, field.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, field.shape[1])
    return field[r_start:r_end, c_start:c_end]

def upscale_region(region, scale_factor=5, order=3):
    """
    Upscale the region using interpolation.
    
    Parameters:
      region (np.ndarray): 2D subregion.
      scale_factor (float): Factor by which to upscale.
      order (int): Interpolation order (3 for cubic, for example).
      
    Returns:
      np.ndarray: Upscaled high-resolution region.
    """
    return zoom(region, zoom=scale_factor, order=order)

def compute_local_gradients(field):
    """
    Compute local gradient magnitude using a Gaussian gradient filter.
    
    Returns:
      tuple: (mean gradient, max gradient)
    """
    # Use gaussian_gradient_magnitude for a robust measure.
    grad_mag = gaussian_gradient_magnitude(field, sigma=1)
    return np.mean(grad_mag), np.max(grad_mag)

def plot_field(field, title="Field", cmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.imshow(field, cmap=cmap, origin='lower')
    plt.title(title)
    plt.colorbar(label="Field Value")
    plt.tight_layout()
    plt.show()

def main():
    # Load final snapshot
    field = load_final_snapshot()
    if field is None:
        print("No final snapshot available.")
        return
    
    # Locate the vortex (assumed to be a bright dot)
    vortex_center = find_dot_center(field, mode='max')
    print(f"Detected vortex center at (row, col): {vortex_center}")
    
    # Extract a zoomed region around the vortex
    window_size = 20  # adjust as needed
    zoom_region = extract_zoom(field, vortex_center, window_size=window_size)
    
    # Upscale the region for fine detail visualization
    scale_factor = 20  # adjust to resolve more details
    high_res_region = upscale_region(zoom_region, scale_factor=scale_factor, order=3)
    
    # Compute local gradient statistics on the original zoom region
    mean_grad, max_grad = compute_local_gradients(zoom_region)
    
    # Print local statistics
    print("Zoomed Region Statistics:")
    print(f"  Window size (original): {zoom_region.shape}")
    print(f"  Upscaled region shape: {high_res_region.shape}")
    print(f"  Mean gradient: {mean_grad:.4e}")
    print(f"  Max gradient: {max_grad:.4e}")
    
    # Plot the original zoomed region
    plot_field(zoom_region, title="Original Zoomed Region Around Vortex")
    
    # Plot the high-resolution zoomed region
    plot_field(high_res_region, title=f"High-Resolution Zoom (scale factor = {scale_factor})")
    
if __name__ == "__main__":
    main()
