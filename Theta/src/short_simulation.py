#!/usr/bin/env python3
"""
short_simulation.py

A short DV-RIPE PDE run that:
  1. Reads parameters from 'scan_config.txt'.
  2. Initializes the PDE field/domain (no toy code: calls real PDE logic).
  3. Runs for a short time (default ~0.5s or as specified by 'run_time' in config).
  4. Saves 'short_final.npy'.

Usage:
    python short_simulation.py

Requires:
    dvripe_physics.py  (must provide real PDE solver routines)
"""

import os
import sys
import numpy as np

# Real PDE solver routines:
from dvripe_physics import initialize_field, run_pde_step

def load_params_from_config(config_file="scan_config.txt") -> dict:
    """
    Reads key-value pairs from config_file and returns them as a dict.
    Expects float-convertible values for PDE parameters like gamma, eta, etc.
    Also checks for optional 'run_time'.
    """
    params = {}
    if not os.path.exists(config_file):
        return params
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return params

def main():
    # 1. Load PDE parameters from config
    params = load_params_from_config("scan_config.txt")

    # 2. Decide domain size or read from config
    Nx = 64
    Ny = 64
    dt = 0.01
    run_time = params.get("run_time", 0.5)  # default 0.5s if not specified
    steps = int(run_time / dt)

    # 3. Initialize the PDE field
    #    Real PDE logic: e.g. pass Nx, Ny, plus PDE param dict
    field = initialize_field(params, Nx, Ny)
    # field is a 2D np.ndarray representing the mass-energy field

    # 4. PDE evolution
    for step in range(steps):
        # run one PDE step
        field = run_pde_step(field, dt, params)
        # if you want occasional logging or intermediate snapshots, do it here

    # 5. Save final field
    np.save("short_final.npy", field)
    print("Short simulation complete. Saved 'short_final.npy'.")

if __name__ == "__main__":
    main()
