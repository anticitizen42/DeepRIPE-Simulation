#!/usr/bin/env python3
"""
slideshow_snapshots.py

This script creates a slideshow from simulation snapshot files.
It assumes that snapshots are saved as NumPy files in the directory:
    data/field_snapshots/
with filenames formatted as "field_<time>.npy" (e.g., field_0.0000.npy).

The script sorts the files by time, loads them, and animates a slideshow of
the absolute field amplitude.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re

def load_snapshot_files(directory="data/field_snapshots"):
    """
    Get a sorted list of snapshot files based on the time extracted from the filename.
    """
    pattern = directory + "/field_*.npy"
    files = glob.glob(pattern)
    # Extract time from filename using a regex and sort by it.
    files_sorted = sorted(files, key=lambda f: float(re.findall(r"field_(\d+\.\d+)", f)[0]))
    return files_sorted

def animate_snapshots(files, interval=500):
    """
    Create an animation slideshow from the snapshot files.
    interval: time between frames in milliseconds.
    """
    # Load first snapshot to set up the figure.
    snapshot = np.load(files[0])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.abs(snapshot), origin="lower", cmap="viridis")
    ax.set_title(f"Time: {re.findall(r'field_(\d+\.\d+)', files[0])[0]}")
    plt.xlabel("Angular Index")
    plt.ylabel("Radial Index")
    plt.colorbar(im, ax=ax, label="|φ|")

    def update(frame):
        snapshot = np.load(files[frame])
        im.set_data(np.abs(snapshot))
        time_val = re.findall(r"field_(\d+\.\d+)", files[frame])[0]
        ax.set_title(f"Time: {time_val}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(files),
                                  interval=interval, blit=True)
    plt.show()

def main():
    files = load_snapshot_files()
    if not files:
        print("No snapshot files found. Please check your data/field_snapshots/ directory.")
        return
    print("Found snapshot files:")
    for f in files:
        print(f)
    animate_snapshots(files)

if __name__ == "__main__":
    main()
