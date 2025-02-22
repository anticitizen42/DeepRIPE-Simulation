# src/fields.py

import numpy as np

def init_electron_fields(shape):
    phi1 = np.zeros(shape, dtype=np.complex128)
    phi2 = np.zeros(shape, dtype=np.complex128)
    # Example: set a small vortex at the center
    center = tuple(s//2 for s in shape)
    phi1[center] = 0.1 + 0.05j
    phi2[center] = 0.08 - 0.03j
    return phi1, phi2

def init_gauge_field(shape):
    # shape might be (4, Nx, Ny, Nz)
    A = np.zeros(shape, dtype=np.complex128)
    return A

def init_gravity_field(shape):
    # store a single scalar potential
    Phi = np.zeros(shape, dtype=np.float64)
    return Phi
