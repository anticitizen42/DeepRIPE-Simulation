#!/usr/bin/env python3
"""
src/polar_analysis.py
Version 1.2

This module provides tools for analyzing a 2D field in polar coordinates.
It includes functions to:
  - Convert a Cartesian 2D field to polar coordinates.
  - Compute refined local angular autocorrelation diagnostics for each radial slice.
    Specifically, it fits the angular autocorrelation to an exponential decay,
    A*exp(-lag/τ), and extracts the decay constant τ as a refined correlation length.
  - Plot the polar-mapped field, angular autocorrelation profiles for selected radii,
    and the decay constant (τ) versus radius.
  - Save the refined autocorrelation data to a text file for further interpretation.
  
These tools help you dynamically map suspected vortex structures and may provide insight
into which parameters might be missing or need adjustment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def cart2polar(field, center=None, final_radius=None, radial_steps=200, angular_steps=360):
    """
    Convert a 2D Cartesian field into polar coordinates.
    
    Parameters:
      field (np.ndarray): 2D array (shape: (Ny, Nx)).
      center (tuple): (x_center, y_center). Defaults to the field's center.
      final_radius (float): Maximum radius to consider. Defaults to distance from center to a corner.
      radial_steps (int): Number of radial samples.
      angular_steps (int): Number of angular samples.
      
    Returns:
      r_grid (np.ndarray): 2D array of radial coordinates.
      theta_grid (np.ndarray): 2D array of angular coordinates.
      polar_field (np.ndarray): Field values on the polar grid.
    """
    Ny, Nx = field.shape
    if center is None:
        center = (Nx / 2, Ny / 2)
    if final_radius is None:
        final_radius = np.sqrt((max(center[0], Nx - center[0]))**2 +
                               (max(center[1], Ny - center[1]))**2)
    
    r = np.linspace(0, final_radius, radial_steps)
    theta = np.linspace(0, 2*np.pi, angular_steps, endpoint=False)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    
    Xc, Yc = center
    X = Xc + r_grid * np.cos(theta_grid)
    Y = Yc + r_grid * np.sin(theta_grid)
    
    polar_field = map_coordinates(field, [Y.ravel(), X.ravel()], order=1, mode='reflect')
    polar_field = polar_field.reshape(r_grid.shape)
    
    return r_grid, theta_grid, polar_field

def exponential_decay(lag, tau, A):
    """Exponential decay model: A * exp(-lag/tau)."""
    return A * np.exp(-lag / tau)

def compute_exponential_decay(ac_profile):
    """
    Fit the angular autocorrelation profile (excluding zero lag) to an exponential decay 
    and return the decay constant tau.
    
    Parameters:
      ac_profile (np.ndarray): 1D array of autocorrelation values as a function of angular lag.
      
    Returns:
      tau (float): Decay constant in angular samples. Returns NaN if fitting fails.
    """
    lags = np.arange(1, len(ac_profile))  # Exclude zero lag.
    y = ac_profile[1:]
    # Avoid divide-by-zero in cases where data is essentially constant.
    if np.std(y) < 1e-8:
        return np.nan
    try:
        # Initial guess: tau ~ half the range, and A = first value (should be ~1)
        popt, _ = curve_fit(exponential_decay, lags, y, p0=[len(y)/2, 1.0], maxfev=1000)
        tau = popt[0]
    except Exception:
        tau = np.nan
    return tau

def compute_angular_autocorrelation(polar_field):
    """
    For each radial slice, compute the angular autocorrelation and refine it by fitting
    to an exponential decay to extract a decay constant tau.
    
    Parameters:
      polar_field (np.ndarray): 2D array in polar coordinates (shape: (radial_steps, angular_steps)).
    
    Returns:
      autocorr (np.ndarray): Array of shape (radial_steps, angular_steps) with the autocorrelation profiles.
      tau_values (np.ndarray): Array of decay constants tau (in angular samples) for each radial slice.
    """
    radial_steps, angular_steps = polar_field.shape
    autocorr = np.zeros((radial_steps, angular_steps))
    tau_values = np.zeros(radial_steps)
    
    for i in range(radial_steps):
        signal = polar_field[i, :]
        signal = signal - np.mean(signal)
        fft_signal = np.fft.fft(signal)
        power = np.abs(fft_signal)**2
        ac_full = np.fft.ifft(power).real
        ac = ac_full / ac_full[0]
        autocorr[i, :] = ac
        tau_values[i] = compute_exponential_decay(ac)
    
    return autocorr, tau_values

def plot_polar_field(r_grid, theta_grid, polar_field, title="Field in Polar Coordinates"):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(theta_grid, r_grid, polar_field, shading='auto', cmap='viridis')
    plt.xlabel("Angle (radians)")
    plt.ylabel("Radius")
    plt.title(title)
    plt.colorbar(label="Field Value")
    plt.tight_layout()
    plt.show()

def plot_angular_autocorrelation(r, autocorr_profile, tau, title="Angular Autocorrelation vs. Angle"):
    lags = np.arange(len(autocorr_profile))
    plt.figure(figsize=(6, 4))
    plt.plot(lags, autocorr_profile, marker='o', linestyle='-', label="Data")
    # Plot the fitted exponential if tau is valid.
    if not np.isnan(tau):
        fitted = exponential_decay(lags, tau, 1.0)
        plt.plot(lags, fitted, linestyle='--', label=f"Exp fit, tau={tau:.2f}")
    plt.xlabel("Angular Lag (samples)")
    plt.ylabel("Autocorrelation")
    plt.title(f"{title} (r = {r:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_decay_vs_radius(r_values, tau_values):
    plt.figure(figsize=(6, 4))
    plt.plot(r_values, tau_values, marker='o', linestyle='-')
    plt.xlabel("Radius")
    plt.ylabel("Decay Constant Tau (angular samples)")
    plt.title("Exponential Decay Tau vs. Radius")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Load or generate a test field.
    if os.path.exists("data/final_massenergy_field.npy"):
        field = np.load("data/final_massenergy_field.npy")
    else:
        field = np.random.rand(128, 128).astype(np.float32)
    
    r_grid, theta_grid, polar_field = cart2polar(field, radial_steps=200, angular_steps=360)
    plot_polar_field(r_grid, theta_grid, polar_field, title="Mass–Energy Field in Polar Coordinates")
    
    autocorr, tau_values = compute_angular_autocorrelation(polar_field)
    
    # Print the refined tau values.
    print("Refined angular decay constants (tau) for each radial slice (in angular samples):")
    for i, tau in enumerate(tau_values):
        print(f"Radius index {i}: tau = {tau}")
    
    # Plot autocorrelation for a selected radius (e.g., mid-range).
    mid_index = polar_field.shape[0] // 2
    sample_r = r_grid[mid_index, mid_index]
    plot_angular_autocorrelation(sample_r, autocorr[mid_index, :], tau_values[mid_index],
                                 title="Angular Autocorrelation vs. Angle")
    
    r_values = np.linspace(0, np.max(r_grid), polar_field.shape[0])
    plot_decay_vs_radius(r_values, tau_values)

if __name__ == "__main__":
    main()
