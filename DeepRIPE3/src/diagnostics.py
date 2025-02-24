#!/usr/bin/env python3
"""
src/diagnostics.py

Enhanced Diagnostics for the DV-RIPE Simulation.

This module provides functions to compute diagnostic quantities from the final fields:
  - compute_spin: Estimates effective spin via phase winding of the electron field.
  - compute_charge: Computes a surrogate net charge from the gauge field,
      handling both Cartesian (6D) and polar (5D) representations.
  - compute_gravity_indentation: Estimates an energy proxy from gravitational potential differences.
  - compute_energy_flux: Estimates the flux of energy across the field.
  - compute_phase_discontinuity: Quantifies phase jumps in the electron field.
  - compute_interference_pattern: Uses FFT to analyze dominant frequencies in the phase.
  - compute_subharmonic_modes: Identifies subharmonic modes (for polar grids) that may relate to resonant behavior.
  
Additional diagnostics can be added as needed to probe the emergent vortex dynamics.
"""

import numpy as np
from numpy.fft import fft, fftshift

def compute_spin(phi, dx):
    """
    Compute effective spin from the winding of the electron field.
    
    Parameters:
      phi (ndarray): Complex electron field.
      dx (float): Grid spacing.
      
    Returns:
      float: Effective spin (half the winding number, with sign).
    """
    phase = np.angle(phi)
    dphase = np.gradient(phase, dx, axis=-1)
    winding = np.sum(dphase) * dx / (2*np.pi)
    return -0.5 * winding

def compute_charge(A, dx):
    """
    Compute a net charge surrogate from the gauge field A.
    
    Parameters:
      A (ndarray): Gauge field array.
          - For Cartesian mode: expected 6D shape, e.g. (3,4,N0,N1,Ny,Nz).
          - For polar mode: expected 5D shape, e.g. (3,4,N0,N1,Nr).
      dx (float): Grid spacing.
      
    Returns:
      float: Net charge computed via divergence of a representative component.
    """
    # For Cartesian mode: use group 0, component 0.
    if A.ndim == 6:
        comp = A[0, 0]  # shape (N0, N1, Ny, Nz)
    # For polar mode: assume gauge field averaged over the angular coordinate (shape (3,4,N0,N1,Nr))
    elif A.ndim == 5:
        # Use a representative slice along the radial dimension.
        comp = A[0, 0]  # shape (N0, N1, Nr)
        # Further reduce it by averaging over the radial axis:
        comp = np.mean(comp, axis=-1)  # shape (N0, N1)
    elif A.ndim == 4:
        comp = A[0]
    else:
        raise ValueError("Unexpected gauge field dimensions: " + str(A.shape))
    
    # Compute gradient along spatial axes.
    grad_list = np.gradient(comp.real, dx)
    divergence = sum(np.sum(g) for g in grad_list)
    return divergence

def compute_gravity_indentation(Phi, dx, mass_scale=1.0):
    """
    Compute gravitational indentation as a surrogate energy proxy.
    
    Parameters:
      Phi (ndarray): Gravitational potential field.
      dx (float): Grid spacing.
      mass_scale (float): Scaling factor for mass.
      
    Returns:
      float: Gravitational indentation.
    """
    inner = Phi[1:-1,1:-1,1:-1]
    indentation = (inner.max() - inner.min()) * mass_scale
    return indentation

def compute_energy_flux(phi, dx):
    """
    Compute an estimate of the energy flux across the boundaries of the electron field.
    
    Parameters:
      phi (ndarray): Complex electron field.
      dx (float): Grid spacing.
      
    Returns:
      float: Surrogate energy flux.
    """
    energy_density = np.abs(phi)**2
    grad_x = np.gradient(energy_density, dx, axis=-1)
    grad_y = np.gradient(energy_density, dx, axis=-2)
    if energy_density.ndim >= 3:
        grad_z = np.gradient(energy_density, dx, axis=-3)
    else:
        grad_z = 0
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + (grad_z**2 if isinstance(grad_z, np.ndarray) else 0))
    flux = np.sum(grad_mag) * (dx**energy_density.ndim)
    return flux

def compute_phase_discontinuity(phi):
    """
    Compute a measure of phase discontinuity in the electron field.
    
    Parameters:
      phi (ndarray): Complex electron field.
      
    Returns:
      dict: {max_jump, mean_jump, std_jump}
    """
    phase = np.angle(phi)
    diff_x = np.abs(np.diff(phase, axis=-1))
    diff_y = np.abs(np.diff(phase, axis=-2))
    if phase.ndim >= 3:
        diff_z = np.abs(np.diff(phase, axis=-3))
        all_diffs = np.concatenate([diff_x.flatten(), diff_y.flatten(), diff_z.flatten()])
    else:
        all_diffs = np.concatenate([diff_x.flatten(), diff_y.flatten()])
    return {"max_jump": float(np.max(all_diffs)),
            "mean_jump": float(np.mean(all_diffs)),
            "std_jump": float(np.std(all_diffs))}

def compute_interference_pattern(phi):
    """
    Analyze the interference pattern in the electron field via FFT.
    
    Parameters:
      phi (ndarray): Complex electron field.
      
    Returns:
      dict: Contains dominant frequency index and phase variance.
    """
    phase = np.angle(phi)
    # For multi-dimensional fields, take a central slice.
    if phi.ndim >= 3:
        slice_phase = phase[phase.shape[0]//2]
    else:
        slice_phase = phase
    fft_phase = fftshift(fft(slice_phase))
    fft_magnitude = np.abs(fft_phase)
    dominant_idx = np.unravel_index(np.argmax(fft_magnitude), fft_magnitude.shape)
    return {"dominant_freq": dominant_idx,
            "phase_variance": float(np.var(phase))}

def compute_subharmonic_modes(phi):
    """
    Identify subharmonic modes in the electron field by performing an FFT along the angular dimension.
    This function is intended for fields defined on a polar grid.
    
    Parameters:
      phi (ndarray): Complex electron field on a polar grid.
      
    Returns:
      dict: Dominant subharmonic modes and their amplitudes.
    """
    phase = np.angle(phi)
    # Assume the last axis corresponds to the angular coordinate.
    fft_angles = np.fft.fft(phase, axis=-1)
    amplitudes = np.abs(fft_angles)
    flat_amp = amplitudes.flatten()
    indices = np.argpartition(flat_amp, -3)[-3:]
    indices = indices[np.argsort(-flat_amp[indices])]
    freqs = [np.unravel_index(i, amplitudes.shape) for i in indices]
    return {"subharmonic_modes": freqs,
            "amplitudes": flat_amp[indices].tolist()}

if __name__ == "__main__":
    # For testing, we generate a dummy vortex field in Cartesian coordinates.
    Nx, Ny = 64, 64
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2) + 1e-6
    theta = np.arctan2(Y, X)
    # Generate a vortex with a 720-degree phase evolution.
    phi_dummy = (1/R) * np.exp(1j * (2 * theta))
    
    print("Spin:", compute_spin(phi_dummy, dx=(x[1]-x[0])))
    print("Charge (Cartesian):", compute_charge(np.expand_dims(np.expand_dims(phi_dummy,0),0), dx=(x[1]-x[0])))
    print("Gravity Indentation (dummy):", compute_gravity_indentation(np.zeros((Nx,Ny,Nx)), dx=(x[1]-x[0])))
    print("Energy flux:", compute_energy_flux(phi_dummy, dx=(x[1]-x[0])))
    print("Phase discontinuity:", compute_phase_discontinuity(phi_dummy))
    print("Interference pattern:", compute_interference_pattern(phi_dummy))
    print("Subharmonic modes (if polar):", compute_subharmonic_modes(phi_dummy))
