#!/usr/bin/env python3
"""
analyze_double_zoom.py

This script analyzes and visualizes the DV‑RIPE mass–energy field by creating a double‐zoom:
  - A zoomed-out view of the overall field with a rectangle marking the region of interest.
  - A zoomed-in view of the high‐energy nexus.
It then prints the sizes (in pixels) of the zoom windows to help derive a scale.

Usage:
    python analyze_double_zoom.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    Here we assume the nexus is the point of maximum absolute amplitude.
    Returns the (row, col) indices.
    """
    # Compute amplitude.
    amplitude = np.abs(field)
    idx = np.argmax(amplitude)
    nexus_coords = np.unravel_index(idx, field.shape)
    print(f"Identified nexus at (row, col) = {nexus_coords}")
    return nexus_coords

def extract_zoom(field, center, window_size):
    """
    Extract a zoomed region around a center.
    center: (row, col)
    window_size: half-size (in pixels) of the square region.
    Returns the zoomed region and the indices used.
    """
    row, col = center
    r_start = max(row - window_size, 0)
    r_end = min(row + window_size, field.shape[0])
    c_start = max(col - window_size, 0)
    c_end = min(col + window_size, field.shape[1])
    zoom_region = field[r_start:r_end, c_start:c_end]
    return zoom_region, (r_start, r_end, c_start, c_end)

def plot_double_zoom(field, nexus_coords, zoom_in_size=20, zoom_out_size=60):
    """
    Create a double-zoom view:
      - Global view with a rectangle indicating the zoom_in region.
      - A zoomed-out view around the nexus.
      - A tight zoomed-in view.
    """
    # Global view.
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(field), origin="lower", cmap="viridis")
    plt.title("Global Field Amplitude")
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")

    # Draw a rectangle for the zoom-in region.
    r0, c0 = nexus_coords
    # Use zoom_in_size as half-size.
    x = max(c0 - zoom_in_size, 0)
    y = max(r0 - zoom_in_size, 0)
    width = min(2 * zoom_in_size, field.shape[1] - x)
    height = min(2 * zoom_in_size, field.shape[0] - y)
    rect = Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

    # Zoomed-out view: center at nexus, larger window.
    zoom_out_region, bounds_out = extract_zoom(field, nexus_coords, zoom_out_size)
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(zoom_out_region), origin="lower", cmap="viridis")
    plt.title(f"Zoom-Out (±{zoom_out_size} pixels)")
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")

    # Zoomed-in view: center at nexus, smaller window.
    zoom_in_region, bounds_in = extract_zoom(field, nexus_coords, zoom_in_size)
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(zoom_in_region), origin="lower", cmap="viridis")
    plt.title(f"Zoom-In (±{zoom_in_size} pixels)")
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")

    plt.tight_layout()
    plt.show()

    # Print scales.
    print(f"Global field size: {field.shape}")
    print(f"Zoom-out region bounds: {bounds_out} (size: {(bounds_out[1]-bounds_out[0], bounds_out[3]-bounds_out[2])})")
    print(f"Zoom-in region bounds: {bounds_in} (size: {(bounds_in[1]-bounds_in[0], bounds_in[3]-bounds_in[2])})")

def main():
    field = load_field()
    nexus_coords = find_nexus(field)
    plot_double_zoom(field, nexus_coords, zoom_in_size=20, zoom_out_size=60)

if __name__ == "__main__":
    main()
