# src/fields.py

import numpy as np

def init_electron_fields(shape, phase_offset=0.1):
    """
    Initialize two nearly identical electron fields (phi1 and phi2) with a robust vortex profile.
    
    Assumes `shape` is (N0, N1, Ny, Nz), where the vortex is defined in the (Ny, Nz) plane.
    The vortex is given by:
         amplitude f(r) = r / sqrt(r^2 + r0^2)
         phase theta = arctan2(Z, Y)
    so that φ = f(r) exp(i θ), ensuring a zero at the core and a 2π winding.
    
    For resonant dynamics, phi2 is seeded with a small phase offset.
    
    Parameters:
      shape: tuple (N0, N1, Ny, Nz)
      phase_offset: float, phase offset (in radians) for phi2 relative to phi1
      
    Returns:
      phi1, phi2: two complex arrays with the vortex profile
    """
    N0, N1, Ny, Nz = shape
    # Create a grid for the vortex in the (Ny, Nz) plane.
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    r = np.sqrt(Y**2 + Z**2)
    r0 = 0.1  # smoothing parameter for the vortex core
    amplitude = r / np.sqrt(r**2 + r0**2)
    theta = np.arctan2(Z, Y)
    
    vortex = amplitude * np.exp(1j * theta)
    
    # Seed the fields by replicating the vortex across N0 and N1.
    phi1 = np.zeros(shape, dtype=np.complex128)
    phi2 = np.zeros(shape, dtype=np.complex128)
    for i in range(N0):
        for j in range(N1):
            phi1[i, j, :, :] = vortex
            # Introduce a small phase offset in phi2 to trigger resonance.
            phi2[i, j, :, :] = vortex * np.exp(1j * phase_offset)
    
    return phi1, phi2

def init_gauge_field(shape, noise_level=0.01):
    """
    Initialize the gauge field with a configuration that mirrors the vortex structure.
    
    Assumes shape = (4, Nx, Ny, Nz), where the vortex is embedded in the (Ny, Nz) plane.
    Here, we compute the phase gradient of a standard vortex (in the (Ny, Nz) plane) to
    set the spatial components of the gauge field. Small random noise is added to help
    trigger dynamic responses.
    
    Parameters:
      shape: tuple (4, Nx, Ny, Nz)
      noise_level: float, amplitude of the added random noise
      
    Returns:
      A: complex array of shape (4, Nx, Ny, Nz)
    """
    # shape: (4, Nx, Ny, Nz)
    _, Nx, Ny, Nz = shape
    A = np.zeros(shape, dtype=np.complex128)
    
    # Define a grid in the (Ny, Nz) plane.
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    theta = np.arctan2(Z, Y)
    
    # Compute the phase gradients in the (Ny, Nz) plane using central differences.
    dtheta_dy, dtheta_dz = np.gradient(theta, y, z, edge_order=2)
    
    # Replicate the gradients along the x-direction (assumed to be the second spatial axis).
    for ix in range(Nx):
        A[2, ix, :, :] = dtheta_dy + noise_level * (np.random.rand(Ny, Nz) - 0.5)
        A[3, ix, :, :] = dtheta_dz + noise_level * (np.random.rand(Ny, Nz) - 0.5)
    # The time component (A[0]) and an additional spatial component (A[1]) remain zero.
    return A

def init_gravity_field(shape):
    """
    Initialize the gravitational field as a trivial scalar potential.
    
    Parameters:
      shape: tuple for the gravity field grid (e.g. (Nx, Ny, Nz))
      
    Returns:
      Phi: float array of shape 'shape'
    """
    Phi = np.zeros(shape, dtype=np.float64)
    return Phi
