# src/fields.py

import numpy as np

def init_electron_fields(shape, ambient=False, phase_offset=0.1):
    """
    Initialize two electron fields (phi1 and phi2) with either ambient random noise
    or a robust vortex profile.

    Parameters:
      shape: tuple (N0, N1, Ny, Nz)
      ambient: bool. If True, the fields are seeded with small random fluctuations.
               If False, a vortex is seeded in the (Ny, Nz) plane.
      phase_offset: float. If using a vortex profile, phi2 receives an additional phase offset.

    Returns:
      phi1, phi2: two complex NumPy arrays of shape 'shape'
    """
    N0, N1, Ny, Nz = shape
    phi1 = np.zeros(shape, dtype=np.complex128)
    phi2 = np.zeros(shape, dtype=np.complex128)
    
    if ambient:
        # Seed with small random noise (ambient fluctuations)
        noise_level = 0.05
        phi1 = noise_level * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
        phi2 = noise_level * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    else:
        # Seed a robust vortex in the (Ny, Nz) plane.
        # Create coordinates in the vortex plane.
        y = np.linspace(-1, 1, Ny)
        z = np.linspace(-1, 1, Nz)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        r = np.sqrt(Y**2 + Z**2)
        r0 = 0.1  # smoothing parameter to avoid singularity at r=0
        amplitude = r / np.sqrt(r**2 + r0**2)
        theta = np.arctan2(Z, Y)
        vortex = amplitude * np.exp(1j * theta)
        # Replicate the vortex along the first two dimensions.
        for i in range(N0):
            for j in range(N1):
                phi1[i, j, :, :] = vortex
                phi2[i, j, :, :] = vortex * np.exp(1j * phase_offset)
    return phi1, phi2

def init_gauge_field(shape, ambient=False, noise_level=0.01):
    """
    Initialize the gauge field A.

    Parameters:
      shape: tuple (4, Nx, Ny, Nz)
      ambient: bool. If True, the gauge field is seeded with small random noise.
               If False, the gauge field is seeded based on the phase gradient of a vortex.
      noise_level: float. The amplitude of noise to add.

    Returns:
      A: complex NumPy array of shape 'shape'
    """
    # shape: (4, Nx, Ny, Nz)
    _, Nx, Ny, Nz = shape
    A = np.zeros(shape, dtype=np.complex128)
    
    if ambient:
        # Seed with small random fluctuations.
        A = noise_level * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    else:
        # Seed gauge field to reflect the vortex phase gradient.
        # Define a grid in the (Ny, Nz) plane.
        y = np.linspace(-1, 1, Ny)
        z = np.linspace(-1, 1, Nz)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        theta = np.arctan2(Z, Y)
        # Compute central differences for dtheta/dy and dtheta/dz.
        dtheta_dy, dtheta_dz = np.gradient(theta, y, z, edge_order=2)
        # Replicate these gradients along the x-direction (Nx dimension).
        for ix in range(Nx):
            A[2, ix, :, :] = dtheta_dy + noise_level * (np.random.rand(Ny, Nz) - 0.5)
            A[3, ix, :, :] = dtheta_dz + noise_level * (np.random.rand(Ny, Nz) - 0.5)
        # Leave A[0] (time component) and A[1] as zero.
    return A

def init_gravity_field(shape):
    """
    Initialize the gravitational field as a scalar potential.

    Parameters:
      shape: tuple (Nx, Ny, Nz) for the gravity grid.

    Returns:
      Phi: NumPy array of shape 'shape' (float64).
    """
    Phi = np.zeros(shape, dtype=np.float64)
    return Phi
