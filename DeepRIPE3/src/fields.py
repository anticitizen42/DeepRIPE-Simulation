#!/usr/bin/env python3
"""
src/fields.py

Field Initialization for DV‑RIPE Simulation

This module provides functions for initializing the fields required for the DV‑RIPE simulation:
  - init_electron_fields: Initializes the electron (scalar) fields.
       • Supports both Cartesian and polar discretization.
       • In polar mode, the electron field is generated on a polar grid defined by:
           (N0, N1) grid points, with additional radial (Nr) and angular (Nθ) resolution.
  - init_nonabelian_gauge_field: Initializes a non‑Abelian (SU(2)) gauge field.
       • In polar mode, the gauge field shape should be provided as 
         (3, 4, N0, N1, Nr, Nθ). It can be seeded with ambient noise.
  - init_gravity_field: Initializes the gravitational potential field as a zero‑field.
"""

import numpy as np

def init_electron_fields(shape, ambient=False, polar=False, phase_offset=0.1,
                         radial_points=64, angular_points=64, r_max=1.0):
    """
    Initialize the electron (scalar) fields φ1 and φ2.
    
    Parameters:
      shape (tuple): For Cartesian mode, shape = (N0, N1, Ny, Nz). For polar mode,
                     only the first two dimensions (N0, N1) are used, and the actual grid
                     will be generated using polar coordinates with radial_points and angular_points.
      ambient (bool): If True, initialize with ambient noise.
      polar (bool):   If True, initialize on a polar grid.
      phase_offset (float): Phase offset for φ2 when seeding a vortex.
      radial_points (int): Number of points in the radial direction (used in polar mode).
      angular_points (int): Number of points in the angular direction (used in polar mode).
      r_max (float): Maximum radial coordinate (defines the spatial extent in polar mode).
      
    Returns:
      tuple: (φ1, φ2) as complex NumPy arrays.
      
    In polar mode, the returned arrays have shape (N0, N1, Nr, Nθ).
    """
    if polar:
        N0, N1 = shape
        # Create radial and angular grids.
        r = np.linspace(0, r_max, radial_points)
        theta = np.linspace(0, 2*np.pi, angular_points, endpoint=False)
        R, Theta = np.meshgrid(r, theta, indexing='ij')  # R: (Nr, Nθ)
        # Generate a vortex: amplitude could be a function of radius.
        vortex = (R / (R + 0.1)) * np.exp(1j * Theta)
        # Initialize φ1 and φ2 for each (N0, N1) grid cell.
        phi1 = np.empty((N0, N1, radial_points, angular_points), dtype=np.complex128)
        phi2 = np.empty((N0, N1, radial_points, angular_points), dtype=np.complex128)
        if ambient:
            noise_level = 0.1
            for i in range(N0):
                for j in range(N1):
                    # Start with ambient noise and add the vortex structure.
                    phi1[i,j] = noise_level * (np.random.randn(radial_points, angular_points) + 1j*np.random.randn(radial_points, angular_points)) + vortex
                    phi2[i,j] = noise_level * (np.random.randn(radial_points, angular_points) + 1j*np.random.randn(radial_points, angular_points)) + vortex * np.exp(1j*phase_offset)
        else:
            phi1.fill(0)
            phi2.fill(0)
            # Optionally, you can seed a pure vortex configuration here.
            for i in range(N0):
                for j in range(N1):
                    phi1[i,j] = vortex
                    phi2[i,j] = vortex * np.exp(1j*phase_offset)
        return phi1, phi2
    else:
        # Cartesian mode initialization.
        N0, N1, Ny, Nz = shape
        if ambient:
            noise_level = 0.1
            phi1 = noise_level * (np.random.randn(N0, N1, Ny, Nz) + 1j*np.random.randn(N0, N1, Ny, Nz))
            phi2 = noise_level * (np.random.randn(N0, N1, Ny, Nz) + 1j*np.random.randn(N0, N1, Ny, Nz))
        else:
            phi1 = np.zeros((N0, N1, Ny, Nz), dtype=np.complex128)
            phi2 = np.zeros((N0, N1, Ny, Nz), dtype=np.complex128)
        return phi1, phi2

def init_nonabelian_gauge_field(shape, ambient=True, noise_level=0.02):
    """
    Initialize a non‑Abelian gauge field for SU(2).
    
    Parameters:
      shape (tuple): For Cartesian mode, typically (3, 4, Nx, Ny, Nz, Nt) or in polar mode,
                     (3, 4, N0, N1, Nr, Nθ).
      ambient (bool): If True, seed with ambient noise.
      noise_level (float): Amplitude of the noise.
      
    Returns:
      ndarray: A complex NumPy array representing the gauge field.
    """
    if ambient:
        A = noise_level * (np.random.randn(*shape) + 1j*np.random.randn(*shape))
    else:
        A = np.zeros(shape, dtype=np.complex128)
    return A

def init_gravity_field(shape):
    """
    Initialize the gravitational potential field.
    
    Parameters:
      shape (tuple): Dimensions of the gravity grid, e.g. (Nx, Ny, Nz).
      
    Returns:
      ndarray: A real NumPy array of zeros.
    """
    return np.zeros(shape, dtype=np.float64)

if __name__ == "__main__":
    # Example usage.
    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=logging_format)
    
    # Test polar mode.
    polar_shape = (16, 32)  # Use only first two dimensions for electron grid.
    phi1, phi2 = init_electron_fields(polar_shape, ambient=True, polar=True, radial_points=64, angular_points=64, r_max=1.0)
    logging.info("Polar electron fields shape: %s, %s", phi1.shape, phi2.shape)
    
    # Test gauge field initialization in polar mode.
    gauge_shape = (3, 4, polar_shape[0], polar_shape[1], 64, 64)
    A = init_nonabelian_gauge_field(gauge_shape, ambient=True)
    logging.info("Gauge field shape (polar): %s", A.shape)
    
    # Test gravity field.
    Phi = init_gravity_field((64, 64, 64))
    logging.info("Gravity field shape: %s", Phi.shape)
