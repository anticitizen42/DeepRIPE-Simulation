# src/fields.py

import numpy as np

def init_electron_fields(shape):
    """
    Initialize electron fields with a robust vortex profile.
    
    Assumes `shape` is (N0, N1, Ny, Nz), where the vortex is formed in the (Ny, Nz) plane.
    We use a profile:
         amplitude f(r) = r / sqrt(r^2 + r0^2)
         phase = theta
    so that φ = f(r) exp(i θ), ensuring a zero at r = 0 and a full 2π winding.
    This vortex profile is replicated uniformly along the N0 and N1 directions.
    """
    N0, N1, Ny, Nz = shape
    # Create a grid in the vortex plane
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    r = np.sqrt(Y**2 + Z**2)
    r0 = 0.1  # small constant to smooth the core
    amplitude = r / np.sqrt(r**2 + r0**2)
    theta = np.arctan2(Z, Y)
    vortex = amplitude * np.exp(1j * theta)
    
    # Initialize electron fields and replicate the vortex profile
    phi1 = np.zeros(shape, dtype=np.complex128)
    phi2 = np.zeros(shape, dtype=np.complex128)
    for i in range(N0):
        for j in range(N1):
            phi1[i, j, :, :] = vortex
            phi2[i, j, :, :] = vortex
    return phi1, phi2

def init_gauge_field(shape):
    """
    Initialize the gauge field with a vortex-like configuration.
    
    Assumes shape = (4, Nx, Ny, Nz), where Nx corresponds to the x-direction,
    and (Ny, Nz) is the plane where the vortex is embedded.
    
    We set:
      - A[0] = 0 (time component)
      - A[1] = 0 (x component)
      - A[2] = dθ/dy (y component)
      - A[3] = dθ/dz (z component)
      
    The gradients dθ/dy and dθ/dz are computed on a grid in the (Ny, Nz) plane,
    and then replicated along the x-direction.
    """
    # shape: (4, Nx, Ny, Nz)
    _, Nx, Ny, Nz = shape
    A = np.zeros(shape, dtype=np.complex128)
    
    # Define a grid for the (Ny, Nz) vortex plane
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    theta = np.arctan2(Z, Y)
    
    # Compute gradients of theta with respect to y and z using central differences
    dtheta_dy, dtheta_dz = np.gradient(theta, y, z, edge_order=2)
    
    # Replicate these gradients along the x-direction (dimension 1)
    for ix in range(Nx):
        A[2, ix, :, :] = dtheta_dy  # y-component
        A[3, ix, :, :] = dtheta_dz  # z-component
    # A[0] and A[1] remain zero.
    return A

def init_gravity_field(shape):
    """
    Initialize the gravitational field as a scalar potential.
    """
    Phi = np.zeros(shape, dtype=np.float64)
    return Phi
