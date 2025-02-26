#!/usr/bin/env python3
"""
drivers/read_snapshots.py
Version 1.0

This script reads all field snapshots stored in the "data/field_snapshots/" folder,
extracts the simulation time from the filenames, computes basic diagnostics (mean and standard deviation),
and plots the evolution of these quantities over time.
It also displays a grid of selected snapshots for visual inspection.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def extract_time_from_filename(filename):
    """
    Given a filename of the form '.../field_<time>.npy', extract the simulation time as a float.
    """
    base = os.path.basename(filename)
    # Expecting a pattern like field_0.0050.npy
    try:
        time_str = base.replace("field_", "").replace(".npy", "")
        return float(time_str)
    except Exception:
        return None

def load_snapshots(snapshots_folder="data/field_snapshots"):
    """
    Load all snapshots from the specified folder and sort them by simulation time.
    
    Returns:
      times (list): Sorted list of simulation times.
      snapshots (list): List of corresponding field snapshots.
    """
    pattern = os.path.join(snapshots_folder, "field_*.npy")
    files = glob.glob(pattern)
    # Extract time from filenames.
    snapshots_data = []
    for f in files:
        t = extract_time_from_filename(f)
        if t is not None:
            snapshots_data.append((t, f))
    # Sort by time.
    snapshots_data.sort(key=lambda x: x[0])
    times = [t for t, _ in snapshots_data]
    snapshots = [np.load(f) for _, f in snapshots_data]
    return times, snapshots

def compute_snapshot_statistics(snapshots):
    """
    Compute basic statistics (mean and std) for each snapshot.
    
    Returns:
      means (list): Mean value for each snapshot.
      stds (list): Standard deviation for each snapshot.
    """
    means = [np.mean(s) for s in snapshots]
    stds = [np.std(s) for s in snapshots]
    return means, stds

def plot_statistics(times, means, stds):
    """
    Plot the evolution of the mean field and standard deviation over time.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(times, means, marker='o', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Mean Field")
    plt.title("Mean Field Evolution")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(times, stds, marker='o', linestyle='-', color='orange')
    plt.xlabel("Time")
    plt.ylabel("Standard Deviation")
    plt.title("Field Variability Evolution")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_snapshot_grid(snapshots, times, num_cols=4):
    """
    Plot a grid of selected snapshots with their simulation times as titles.
    If there are more snapshots than can be evenly arranged, the grid will show as many as possible.
    """
    num_snapshots = len(snapshots)
    num_rows = int(np.ceil(num_snapshots / num_cols))
    
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for i, (t, snap) in enumerate(zip(times, snapshots)):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(snap, cmap='viridis', origin='lower')
        plt.title(f"t = {t:.3f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    snapshots_folder = "data/field_snapshots"
    if not os.path.exists(snapshots_folder):
        print(f"Snapshots folder '{snapshots_folder}' not found.")
        return
    
    times, snapshots = load_snapshots(snapshots_folder)
    if not times:
        print("No snapshot files found.")
        return
    
    print("Loaded Snapshots:")
    for t in times:
        print(f"  Time: {t}")
    
    means, stds = compute_snapshot_statistics(snapshots)
    print("\nSnapshot Statistics:")
    for t, m, s in zip(times, means, stds):
        print(f"Time: {t:.4f}  Mean: {m:.4f}  Std: {s:.4f}")
    
    # Plot statistics over time.
    plot_statistics(times, means, stds)
    
    # Plot a grid of all snapshots.
    plot_snapshot_grid(snapshots, times, num_cols=4)

if __name__ == "__main__":
    main()
