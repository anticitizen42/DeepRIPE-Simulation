#!/usr/bin/env python3
"""
src/fields.py
Version 1.2

Production module for DV‑RIPE simulation field initialization.
This module initializes:
  - Electron fields (supports both Cartesian and polar modes),
  - Gauge fields for non‑Abelian SU(2) dynamics, and
  - Gravity fields.
For electron fields, a new seed type "vortex_perturbed" adds controlled ambient noise to the vortex pattern.
"""

import numpy as np
import logging

# Set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust level as needed for production
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def initialize_electron_field(shape, mode='polar', seed='vortex', noise_level=0.01, random_state=None):
    """
    Initialize the electron field as a complex scalar field.

    Parameters:
        shape (tuple): For polar: (radial_points, angular_points); for Cartesian: (Nx, Ny)
        mode (str): 'polar' or 'cartesian'.
        seed (str): 'vortex', 'vortex_perturbed', or 'noise'.
        noise_level (float): Noise level to add if seed is 'noise' or 'vortex_perturbed'.
        random_state (int or np.random.Generator, optional): Seed for RNG.

    Returns:
        np.ndarray: Complex-valued electron field.
    """
    logger.info(f"Initializing electron field in {mode} mode with seed '{seed}' and grid shape {shape}.")

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    if mode.lower() == 'polar':
        radial_points, angular_points = shape
        r = np.linspace(0, 1, radial_points)
        theta = np.linspace(0, 2 * np.pi, angular_points, endpoint=False)
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        amplitude = np.exp(-R)
        phase = Theta  # Base vortex phase.
        field = amplitude * np.exp(1j * phase)
        if seed.lower() == 'vortex_perturbed':
            # Add small random perturbations to both amplitude and phase.
            field += (rng.normal(scale=noise_level, size=field.shape) +
                      1j * rng.normal(scale=noise_level, size=field.shape))
        elif seed.lower() == 'noise':
            real_noise = rng.normal(scale=noise_level, size=field.shape)
            imag_noise = rng.normal(scale=noise_level, size=field.shape)
            field = real_noise + 1j * imag_noise
        elif seed.lower() != 'vortex':
            logger.error("Unknown seed type for electron field initialization.")
            raise ValueError("seed must be 'vortex', 'vortex_perturbed', or 'noise'")
    elif mode.lower() == 'cartesian':
        Nx, Ny = shape
        x = np.linspace(-1, 1, Nx)
        y = np.linspace(-1, 1, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        amplitude = np.exp(-R)
        phase = Theta
        field = amplitude * np.exp(1j * phase)
        if seed.lower() == 'vortex_perturbed':
            field += (rng.normal(scale=noise_level, size=field.shape) +
                      1j * rng.normal(scale=noise_level, size=field.shape))
        elif seed.lower() == 'noise':
            real_noise = rng.normal(scale=noise_level, size=field.shape)
            imag_noise = rng.normal(scale=noise_level, size=field.shape)
            field = real_noise + 1j * imag_noise
        elif seed.lower() != 'vortex':
            logger.error("Unknown seed type for electron field initialization.")
            raise ValueError("seed must be 'vortex', 'vortex_perturbed', or 'noise'")
    else:
        logger.error("Unknown mode for electron field initialization.")
        raise ValueError("mode must be 'polar' or 'cartesian'")

    logger.info("Electron field initialization complete.")
    return field

def initialize_gauge_field(grid_shape, group_dim=3, component_dim=3, mode='polar'):
    """
    Initialize the non‑Abelian SU(2) gauge field.

    Parameters:
        grid_shape (tuple): e.g. (N0, N1, radial_points, angular_points) for polar mode.
        group_dim (int): Typically 3 for SU(2).
        component_dim (int): Number of spatial components.
        mode (str): 'polar' or 'cartesian'.

    Returns:
        np.ndarray: Gauge field array.
    """
    logger.info(f"Initializing gauge field in {mode} mode with grid shape {grid_shape}, group_dim={group_dim}, component_dim={component_dim}.")
    shape = (group_dim, component_dim) + grid_shape
    gauge_field = np.zeros(shape, dtype=np.float32)
    gauge_field += np.random.normal(loc=0.0, scale=0.01, size=shape).astype(np.float32)
    logger.info("Gauge field initialization complete.")
    return gauge_field

def initialize_gravity_field(grid_shape, mode='polar'):
    """
    Initialize the gravitational potential field.

    Parameters:
        grid_shape (tuple): e.g. (N0, N1, radial_points, angular_points) for polar mode.
        mode (str): 'polar' or 'cartesian'.

    Returns:
        np.ndarray: Real-valued gravitational potential field.
    """
    logger.info(f"Initializing gravity field in {mode} mode with grid shape {grid_shape}.")
    gravity_field = np.zeros(grid_shape, dtype=np.float32)
    gravity_field += np.random.normal(loc=0.0, scale=0.001, size=gravity_field.shape).astype(np.float32)
    logger.info("Gravity field initialization complete.")
    return gravity_field

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running test harness for fields module.")
    
    # Test electron field: vortex_perturbed in polar mode.
    field1 = initialize_electron_field(shape=(64, 128), mode='polar', seed='vortex_perturbed', noise_level=0.05)
    logger.info(f"Electron field (polar, vortex_perturbed) shape: {field1.shape}")
    
    # Test electron field: vortex in polar mode.
    field2 = initialize_electron_field(shape=(64, 128), mode='polar', seed='vortex')
    logger.info(f"Electron field (polar, vortex) shape: {field2.shape}")
    
    # Test gauge and gravity fields.
    gauge_field = initialize_gauge_field(grid_shape=(32, 32, 64, 128), group_dim=3, component_dim=3, mode='polar')
    logger.info(f"Gauge field shape: {gauge_field.shape}")
    gravity_field = initialize_gravity_field(grid_shape=(32, 32, 64, 128), mode='polar')
    logger.info(f"Gravity field shape: {gravity_field.shape}")
