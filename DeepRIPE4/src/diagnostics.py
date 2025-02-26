#!/usr/bin/env python3
"""
src/diagnostics.py
Version 2.0

Enhanced diagnostics suite for the DV‑RIPE simulation.
In addition to global diagnostics (energy proxy, effective spin, net charge),
this module now provides spatially-resolved diagnostics:
  - Local phase winding maps to visualize regional variations in vortex twist.
  - Energy density maps for detailed analysis.
  - Fourier spectral analysis to detect subtle interference structures.
  
These additional tools will help us capture and quantify the effect of parameter variations
on the emergent vortex dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def compute_energy_density(field):
    """
    Compute the energy density of the field.
    Uses finite differences to compute gradient magnitude squared.
    
    Parameters:
        field (np.ndarray): 2D state array.
    
    Returns:
        np.ndarray: 2D energy density.
    """
    grad_y, grad_x = np.gradient(field)
    return grad_x**2 + grad_y**2

def compute_global_energy(field):
    """
    Compute a global energy proxy by integrating energy density.
    
    Parameters:
        field (np.ndarray): 2D state array.
    
    Returns:
        float: Global energy proxy.
    """
    density = compute_energy_density(field)
    return np.sum(density)

def compute_effective_spin(electron_field):
    """
    Compute effective spin by calculating the global phase winding
    over the angular dimension. (Assumes polar coordinates with angular axis=1.)
    
    Parameters:
        electron_field (np.ndarray): Complex electron field.
    
    Returns:
        float: Effective spin (absolute average winding / (2π)).
    """
    phase = np.angle(electron_field)
    dphase = np.diff(phase, axis=1)
    dphase = (dphase + np.pi) % (2*np.pi) - np.pi  # Correct for wrapping
    total_winding = np.mean(np.sum(dphase, axis=1)) / (2*np.pi)
    return abs(total_winding)

def compute_local_phase_winding(electron_field):
    """
    Compute a local phase winding map for the electron field.
    For each radial coordinate, compute the cumulative phase difference along the angular axis.
    
    Parameters:
        electron_field (np.ndarray): Complex electron field in polar coordinates.
    
    Returns:
        np.ndarray: 2D array of local phase winding values.
    """
    phase = np.angle(electron_field)
    # Compute difference along the angular direction.
    dphase = np.diff(phase, axis=1)
    dphase = (dphase + np.pi) % (2*np.pi) - np.pi  # Correct for wrapping
    # Compute cumulative sum along the angular direction.
    local_winding = np.cumsum(dphase, axis=1)
    # Pad to match original dimensions.
    local_winding = np.hstack((np.zeros((phase.shape[0], 1)), local_winding))
    return local_winding

def plot_energy_density(field, title="Energy Density Map"):
    """
    Plot the energy density of the field.
    
    Parameters:
        field (np.ndarray): 2D state array.
        title (str): Plot title.
    """
    density = compute_energy_density(field)
    plt.figure(figsize=(6,5))
    plt.imshow(density, cmap='viridis', origin='lower')
    plt.colorbar(label="Energy Density")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

def plot_local_phase_winding(electron_field, title="Local Phase Winding Map"):
    """
    Plot the local phase winding map of the electron field.
    
    Parameters:
        electron_field (np.ndarray): Complex electron field in polar coordinates.
        title (str): Plot title.
    """
    winding_map = compute_local_phase_winding(electron_field)
    plt.figure(figsize=(6,5))
    plt.imshow(winding_map, cmap='twilight', origin='lower')
    plt.colorbar(label="Cumulative Phase (radians)")
    plt.title(title)
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")
    plt.tight_layout()
    plt.show()

def compute_fourier_spectrum(field):
    """
    Compute the 2D Fourier transform magnitude of the field.
    
    Parameters:
        field (np.ndarray): 2D state array.
    
    Returns:
        np.ndarray: Fourier magnitude spectrum.
    """
    fft_field = np.fft.fft2(field)
    fft_shift = np.fft.fftshift(fft_field)
    return np.abs(fft_shift)

def plot_fourier_spectrum(field, title="Fourier Spectrum"):
    """
    Plot the Fourier magnitude spectrum of the field.
    
    Parameters:
        field (np.ndarray): 2D state array.
        title (str): Plot title.
    """
    spectrum = compute_fourier_spectrum(field)
    plt.figure(figsize=(6,5))
    plt.imshow(spectrum, cmap='inferno', origin='lower')
    plt.colorbar(label="Magnitude")
    plt.title(title)
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.tight_layout()
    plt.show()

def save_diagnostics(diagnostics, filename="diagnostics.npy"):
    """
    Save the diagnostics dictionary to a file.
    
    Parameters:
        diagnostics (dict): Diagnostic data.
        filename (str): File name for saving diagnostics.
    """
    np.save(filename, diagnostics)
    logging.info(f"Diagnostics saved to {filename}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running enhanced diagnostics test harness.")
    
    # Create a dummy complex electron field in polar coordinates.
    r = np.linspace(0, 1, 64)
    theta = np.linspace(0, 2*np.pi, 128, endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    electron_field = np.exp(-R) * np.exp(1j * Theta)
    
    # Plot local phase winding.
    plot_local_phase_winding(electron_field, title="Test Local Phase Winding")
    
    # Create a dummy real field and plot energy density.
    dummy_field = np.sin(np.linspace(0, 4*np.pi, 64)).reshape(8,8)
    plot_energy_density(dummy_field, title="Test Energy Density")
    
    # Plot Fourier spectrum.
    plot_fourier_spectrum(dummy_field, title="Test Fourier Spectrum")
    
    # Save diagnostics example.
    diagnostics = {
        'effective_spin': compute_effective_spin(electron_field),
        'net_charge': np.sum(np.imag(electron_field)) / electron_field.size,
        'energy_proxy': np.mean(np.abs(dummy_field))
    }
    save_diagnostics(diagnostics, filename="test_diagnostics.npy")
