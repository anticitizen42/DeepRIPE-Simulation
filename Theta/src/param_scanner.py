#!/usr/bin/env python3
"""
param_scanner.py

A parameter scanner that systematically tests multiple DV-RIPE PDE parameter sets
and logs the "energy flux through the vortex nexus" as a metric. The flux is computed
by compute_metric.py, which references real PDE flux code (dvripe_physics.py).

Usage:
    python param_scanner.py
"""

import os
import numpy as np
import subprocess
import time

# Updated parameter ranges for your PDE, focusing on higher damping, gauge coupling, etc.
PARAM_RANGES = {
    "gamma":   [0.05, 0.10, 0.15, 0.20],
    "eta":     [0.5, 1.0, 2.0],
    "e_gauge": [0.05, 0.10, 0.15],
    "delta_e": [0.2, 0.3, 0.4],
    "kappa":   [0.0, 0.1, 0.2],
}

SIM_SCRIPT = "short_simulation.py"   # Quick PDE solver that reads scan_config.txt, runs ~0.5-1.0s
METRIC_SCRIPT = "compute_metric.py"  # Prints energy flux as single float
OUTPUT_LOG = "param_scan_results.csv"

def build_param_list():
    """
    Creates a list of dicts, each with one combination of parameters (grid search).
    """
    param_sets = []
    gamma_vals   = PARAM_RANGES["gamma"]
    eta_vals     = PARAM_RANGES["eta"]
    eg_vals      = PARAM_RANGES["e_gauge"]
    delta_vals   = PARAM_RANGES["delta_e"]
    kappa_vals   = PARAM_RANGES["kappa"]

    for g in gamma_vals:
        for e in eta_vals:
            for eg in eg_vals:
                for d in delta_vals:
                    for k in kappa_vals:
                        p = {
                            "gamma":   float(g),
                            "eta":     float(e),
                            "e_gauge": float(eg),
                            "delta_e": float(d),
                            "kappa":   float(k),
                        }
                        param_sets.append(p)
    return param_sets

def write_config_file(params, filename="scan_config.txt"):
    """
    Writes the key-value pairs to config_file, so short_simulation.py can load them.
    """
    with open(filename, "w") as f:
        for k, v in params.items():
            f.write(f"{k}={v}\n")

def run_short_simulation(params):
    """
    1. Write params to config file.
    2. Run short_simulation.py -> produces 'short_final.npy'.
    3. Run compute_metric.py -> prints flux metric to stdout.
    4. Return that metric as float.
    """
    write_config_file(params, "scan_config.txt")

    # Step 1: run short_simulation
    try:
        subprocess.run(["python", SIM_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: short_simulation.py failed with code {e.returncode}")
        return np.nan

    # Step 2: run compute_metric
    try:
        result = subprocess.run(["python", METRIC_SCRIPT],
                                check=True, capture_output=True, text=True)
        metric_str = result.stdout.strip()
        metric_val = float(metric_str)
    except Exception as e:
        print(f"Error: compute_metric.py failed: {e}")
        return np.nan

    return metric_val

def main():
    param_sets = build_param_list()
    print(f"Scanning {len(param_sets)} parameter sets...")

    # Write CSV header
    with open(OUTPUT_LOG, "w") as f:
        f.write("gamma,eta,e_gauge,delta_e,kappa,flux_metric\n")

    best_metric = float('inf')
    best_params = None

    for i, pset in enumerate(param_sets):
        print(f"\nRunning param set {i+1}/{len(param_sets)}: {pset}")
        flux_val = run_short_simulation(pset)
        print(f"  => Flux Metric = {flux_val:.6f}")

        # Log to CSV
        with open(OUTPUT_LOG, "a") as f:
            f.write(f"{pset['gamma']},{pset['eta']},{pset['e_gauge']},"
                    f"{pset['delta_e']},{pset['kappa']},{flux_val:.6f}\n")

        # Track best
        if flux_val < best_metric:
            best_metric = flux_val
            best_params = pset

    print("\nParameter scan complete.")
    if best_metric < float('inf'):
        print(f"Best metric found = {best_metric:.6f} with params = {best_params}")
    else:
        print("All runs returned NaN. Check short_simulation.py or compute_metric.py for errors.")

if __name__ == "__main__":
    main()
