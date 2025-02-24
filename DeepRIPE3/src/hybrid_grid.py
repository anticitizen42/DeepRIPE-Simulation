#!/usr/bin/env python3
"""
src/hybrid_grid.py

This module defines a function to generate a hybrid grid that transitions seamlessly from
a high-resolution polar grid (near the vortex core) to a coarser Cartesian grid (further away).

Parameters:
  - xc, yc: Floats. The coordinates of the vortex core.
  - r_transition: Float. The radius (from the core) within which a high-resolution polar grid is used.
  - dr, dtheta: Floats. The radial and angular resolutions for the polar grid.
  - x_min, x_max, y_min, y_max: Floats. The bounds of the overall simulation domain.
  - cart_spacing: Float. The spacing for the coarser Cartesian grid in the outer region.

The function returns a tuple (X, Y) where X and Y are 1D arrays of the x and y coordinates of the hybrid grid.
An optional plotting function is provided to visualize the grid.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_hybrid_grid(xc, yc, r_transition, dr, dtheta, x_min, x_max, y_min, y_max, cart_spacing):
    """
    Generate a hybrid grid with a high-resolution polar mesh in the inner region (r <= r_transition)
    and a coarser Cartesian grid in the outer region (r > r_transition). The Cartesian grid covers the entire domain.
    
    Returns:
      X, Y: 1D numpy arrays of the combined x and y coordinates.
    """
    # --- Generate polar grid for r <= r_transition ---
    # Create radial and angular arrays.
    r = np.arange(0, r_transition + dr, dr)
    theta = np.arange(0, 2 * np.pi, dtheta)
    # Create a meshgrid in polar coordinates.
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    # Convert polar coordinates to Cartesian.
    X_polar = xc + R * np.cos(Theta)
    Y_polar = yc + R * np.sin(Theta)
    
    # Flatten the polar grid arrays.
    Xp = X_polar.flatten()
    Yp = Y_polar.flatten()
    
    # --- Generate Cartesian grid for the entire domain ---
    x_cart = np.arange(x_min, x_max + cart_spacing, cart_spacing)
    y_cart = np.arange(y_min, y_max + cart_spacing, cart_spacing)
    X_cart, Y_cart = np.meshgrid(x_cart, y_cart, indexing='ij')
    Xc = X_cart.flatten()
    Yc = Y_cart.flatten()
    
    # --- Merge the grids: use the polar points in the inner region, and the Cartesian points outside ---
    # Compute distance from the vortex core for each Cartesian point.
    r_cart = np.sqrt((Xc - xc)**2 + (Yc - yc)**2)
    
    # Keep only Cartesian points that are outside the polar region (with a slight tolerance to avoid duplicates).
    tol = 1e-8
    outer_idx = np.where(r_cart > (r_transition + tol))[0]
    X_outer = Xc[outer_idx]
    Y_outer = Yc[outer_idx]
    
    # Combine the polar and outer Cartesian points.
    X_total = np.concatenate([Xp, X_outer])
    Y_total = np.concatenate([Yp, Y_outer])
    
    # Optionally, remove duplicate points (by rounding coordinates to a tolerance).
    rounded = np.round(np.vstack((X_total, Y_total)).T, decimals=8)
    unique = np.unique(rounded, axis=0)
    X_unique = unique[:, 0]
    Y_unique = unique[:, 1]
    
    return X_unique, Y_unique

def plot_hybrid_grid(X, Y, title="Hybrid Grid"):
    """
    Plot the hybrid grid points for visualization.
    
    Parameters:
      X, Y: 1D arrays of x and y coordinates.
      title: Title of the plot.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(X, Y, s=5, color='blue', marker='.')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Example parameters:
    xc, yc = 0.0, 0.0       # vortex core at (0, 0)
    r_transition = 5.0      # use polar grid inside radius 5
    dr = 0.1                # radial resolution
    dtheta = np.pi / 180.0  # 1 degree resolution in radians
    # Cartesian domain bounds
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    cart_spacing = 0.5      # coarse Cartesian spacing

    X, Y = generate_hybrid_grid(xc, yc, r_transition, dr, dtheta, x_min, x_max, y_min, y_max, cart_spacing)
    plot_hybrid_grid(X, Y, title="Hybrid Polar-Cartesian Grid")
