#!/usr/bin/env python3
"""
analyze_electron_signatures.py

This script analyzes a collection of simulation snapshots (stored as .npy files)
to identify candidate electron-like signatures in the DV‑RIPE mass–energy field.
The approach is as follows:

  1. For each snapshot, load the field and compute its amplitude.
  2. Find the vortex nexus (the point of maximum amplitude).
  3. Threshold the amplitude (e.g. at 50% of the max) and compute the average
     distance from the nexus for pixels above threshold, yielding a rough "core radius."
  4. If the core radius is below a set threshold (indicating a very tight vortex),
     flag the snapshot as a candidate electron signature.
  5. Summarize and optionally visualize the candidate snapshots.

Usage:
    python analyze_electron_signatures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import glob, re
from scipy.ndimage import distance_transform_edt

def load_snapshot_files(directory="data/field_snapshots"):
    pattern = directory + "/field_*.npy"
    files = glob.glob(pattern)
    files_sorted = sorted(files, key=lambda f: float(re.findall(r"field_(\d+\.\d+)", f)[0]))
    return files_sorted

def analyze_snapshot(filename, threshold_fraction=0.5):
    """
    Load a snapshot and compute a candidate "core radius" metric.
    - threshold_fraction: pixels with amplitude above this fraction of max are considered.
    Returns:
      core_radius: average distance of pixels above threshold from the nexus.
      nexus: (row, col) of maximum amplitude.
      field: the loaded field.
    """
    field = np.load(filename)
    # Ensure field is float (if complex, use absolute amplitude)
    if np.iscomplexobj(field):
        amplitude = np.abs(field)
    else:
        amplitude = field
    # Find nexus: location of maximum amplitude.
    idx = np.argmax(amplitude)
    nexus = np.unravel_index(idx, amplitude.shape)
    max_val = amplitude[nexus]
    
    # Create a binary mask of pixels above a threshold (e.g. 50% of max)
    threshold = threshold_fraction * max_val
    mask = amplitude >= threshold
    
    # Compute the distance transform: for each pixel, the distance to the nearest zero in the inverted mask.
    # But here, we want distances from the nexus.
    # Create an image where each pixel's value is its Euclidean distance from the nexus.
    y_indices, x_indices = np.indices(amplitude.shape)
    distances = np.sqrt((y_indices - nexus[0])**2 + (x_indices - nexus[1])**2)
    
    # Average the distances of pixels that are above threshold.
    if np.any(mask):
        core_radius = np.mean(distances[mask])
    else:
        core_radius = np.nan
    
    return core_radius, nexus, field

def plot_candidate(filename, core_radius, nexus, field):
    """
    Plot the amplitude and phase (if applicable) of the field and mark the nexus.
    """
    plt.figure(figsize=(10, 4))
    
    # Plot amplitude.
    plt.subplot(1, 2, 1)
    if np.iscomplexobj(field):
        amplitude = np.abs(field)
    else:
        amplitude = field
    plt.imshow(amplitude, origin="lower", cmap="viridis")
    plt.colorbar(label="|φ|")
    plt.plot(nexus[1], nexus[0], 'ro', markersize=8, label="Nexus")
    plt.title(f"Amplitude\nCore radius: {core_radius:.2f} pixels")
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")
    plt.legend()
    
    # If field is complex, also plot phase.
    if np.iscomplexobj(field):
        plt.subplot(1, 2, 2)
        phase = np.angle(field)
        plt.imshow(phase, origin="lower", cmap="twilight")
        plt.colorbar(label="Phase (radians)")
        plt.plot(nexus[1], nexus[0], 'ro', markersize=8, label="Nexus")
        plt.title("Phase")
        plt.xlabel("Angular Index")
        plt.ylabel("Radial Index")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    files = load_snapshot_files()
    if not files:
        print("No snapshot files found.")
        return

    candidate_threshold = 5.0  # core radius threshold (in pixels) for candidate electrons
    candidates = []

    print("Analyzing snapshots for electron-like signatures...")
    for f in files:
        try:
            core_radius, nexus, field = analyze_snapshot(f, threshold_fraction=0.5)
            print(f"{f}: core radius = {core_radius:.2f} pixels; nexus at {nexus}")
            if core_radius < candidate_threshold:
                candidates.append((f, core_radius, nexus, field))
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    print(f"\nFound {len(candidates)} candidate snapshots with core radius < {candidate_threshold} pixels.")
    for candidate in candidates:
        print("Candidate:", candidate[0], "core radius:", candidate[1])
        # Optionally, plot candidate.
        plot_candidate(candidate[0], candidate[1], candidate[2], candidate[3])

if __name__ == "__main__":
    main()
