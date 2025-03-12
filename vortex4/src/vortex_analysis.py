#!/usr/bin/env python3
"""
vortex_analysis.py

This script reads the simulation output from 'output.txt', performs polar discretization
to sample the amplitude along a circular path around the vortex core, and computes a
wavelet scalogram of the sampled data to analyze the vortex's resonance features.

Dependencies:
    - numpy
    - matplotlib
    - pywt (PyWavelets)
    - scipy (for interpolation)

Usage:
    python3 vortex_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.interpolate import RegularGridInterpolator

# Simulation grid parameters (must match those in the CUDA simulation)
NX = 256
NY = 256
DX = 0.1

# Load the output data from the simulation
# The file output.txt is expected to have amplitude and phase values for each grid point.
# We will only use the amplitude for this analysis.
data = np.loadtxt("output.txt")

# The data is structured so that each row has NX pairs (amplitude, phase).
# We extract the amplitude values.
amplitude_field = np.zeros((NY, NX))
for j in range(NY):
    row = data[j, :]
    amplitude_field[j, :] = row[0::2]  # Extract every even-indexed element (amplitude)

# Define the vortex center (assumed to be at the center of the grid)
cx = NX / 2.0
cy = NY / 2.0

# Choose a radius at which to sample the amplitude (adjust as needed)
radius = 10.0  # in grid units

# Define angular coordinates for sampling
num_angles = 360
angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

# Sample the amplitude along the circle using interpolation
sampled_amplitude = np.zeros(num_angles)

# Create an interpolator for the amplitude field.
# Note: amplitude_field is indexed as (y, x) with y corresponding to rows.
y = np.arange(NY)
x = np.arange(NX)
interp_func = RegularGridInterpolator((y, x), amplitude_field, method='cubic')

for i, theta in enumerate(angles):
    # Compute the (x, y) coordinates in the grid.
    # The interpolator expects a point in the order [y, x]
    x_coord = cx + radius * np.cos(theta)
    y_coord = cy + radius * np.sin(theta)
    sampled_amplitude[i] = interp_func([y_coord, x_coord])

# Perform Continuous Wavelet Transform (CWT) on the sampled amplitude data
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(sampled_amplitude, scales, 'morl', sampling_period=(2*np.pi/num_angles))

# Plot the wavelet scalogram
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), extent=[0, 360, scales[-1], scales[0]], aspect='auto', cmap='jet')
plt.colorbar(label='Magnitude')
plt.xlabel('Angle (degrees)')
plt.ylabel('Scale')
plt.title('Wavelet Scalogram of Vortex Amplitude along Circular Path')
plt.show()
