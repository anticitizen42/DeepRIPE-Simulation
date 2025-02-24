#!/usr/bin/env python3
"""
src/polar_grid.py

This module implements a polar discretization scheme for regions near vortex cores.
It generates an adaptive polar mesh given:
  - A vortex center (xc, yc)
  - A maximum radius (r_max) within which high resolution is desired.
  - A radial resolution (dr)
  - An angular resolution (dtheta, in radians)

The module returns arrays of the polar coordinates (r, theta) and the corresponding Cartesian
coordinates (X, Y). These coordinates can then be used to remap the simulation fields in the vortex core.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_polar_mesh(xc, yc, r_max, dr, dtheta):
    """
    Generate a polar mesh centered at (xc, yc) with radial resolution dr and angular resolution dtheta.
    
    Parameters:
      xc, yc  : float
                The coordinates of the vortex core center.
      r_max   : float
                The maximum radius for the polar grid.
      dr      : float
                Radial step size.
      dtheta  : float
                Angular step size in radians.
                
    Returns:
      X, Y    : 2D numpy arrays of Cartesian coordinates corresponding to the polar grid.
      R, Theta: 2D numpy arrays of polar coordinates.
    """
    # Generate radial and angular coordinate arrays.
    r = np.arange(0, r_max + dr, dr)
    theta = np.arange(0, 2 * np.pi, dtheta)
    
    # Create a 2D meshgrid in polar coordinates.
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Convert polar coordinates to Cartesian coordinates.
    X = xc + R * np.cos(Theta)
    Y = yc + R * np.sin(Theta)
    
    return X, Y, R, Theta

def plot_mesh(X, Y, title="Polar Mesh"):
    """
    Plot the generated mesh for visualization.
    
    Parameters:
      X, Y : 2D numpy arrays of Cartesian coordinates.
      title: str, optional
             Title for the plot.
    """
    plt.figure(figsize=(6, 6))
    # Plot the mesh lines.
    for i in range(X.shape[0]):
        plt.plot(X[i, :], Y[i, :], 'b-', linewidth=0.5)
    for j in range(Y.shape[1]):
        plt.plot(X[:, j], Y[:, j], 'b-', linewidth=0.5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Example usage: generate and plot a polar grid centered at (0, 0)
    # with r_max=10, dr=0.2, and dtheta=pi/60.
    xc, yc = 0.0, 0.0
    r_max = 10.0
    dr = 0.2
    dtheta = np.pi / 60.0  # 3 degrees resolution
    X, Y, R, Theta = generate_polar_mesh(xc, yc, r_max, dr, dtheta)
    plot_mesh(X, Y, title="Adaptive Polar Mesh Around Vortex Core")
