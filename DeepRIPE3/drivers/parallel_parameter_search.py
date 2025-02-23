#!/usr/bin/env python3
"""
drivers/parallel_parameter_search.py

This script demonstrates how to parallelize DV-RIPE simulation runs using Python's
ProcessPoolExecutor. Each simulation is independent, so we can scan the parameter
space (e.g. varying v_e) concurrently across multiple CPU cores.

Hardware: With a 12-core/24-thread AMD 5900X, this should greatly speed up the
parameter search or troubleshooting runs.
"""

import os
import sys
import json
import logging
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import run_dvripe_sim

def run_simulation_for_v_e(v_e_value, fixed_params):
    """
    Runs a DV-RIPE simulation with a specific v_e value while keeping other parameters fixed.
    Returns a diagnostics dictionary.
    """
    sim_params = fixed_params.copy()
    sim_params["v_e"] = v_e_value
    sim_params["tau_end"] = 5.0  # extended simulation duration

    try:
        # run_dvripe_sim returns (spin, charge, grav, phi1_fin) if available.
        result = run_dvripe_sim(sim_params)
        raw_spin = result[0]
        charge = result[1]
        energy_proxy = result[2]
        effective_spin = -0.5 * raw_spin  # our mapping: raw -1 -> effective 0.5
    except Exception as e:
        logging.error(f"Simulation error for v_e = {v_e_value}: {e}")
        return None

    # Attempt to get additional field diagnostics.
    try:
        phi1_fin = result[3]  # final electron field
    except IndexError:
        phi1_fin = None

    diagnostics = {
        "v_e": v_e_value,
        "raw_spin": raw_spin,
        "effective_spin": effective_spin,
        "charge": charge,
        "energy_proxy": energy_proxy,
    }

    if phi1_fin is not None:
        diagnostics.update(extract_field_stats(phi1_fin))
    else:
        diagnostics.update({
            "amp_mean": 0.0,
            "amp_std": 0.0,
            "phase_mean": 0.0,
            "phase_std": 0.0,
        })
    return diagnostics

def extract_central_slice(field):
    """Extract the central 2D slice from a 4D field (N0, N1, Ny, Nz)."""
    N0, N1, Ny, Nz = field.shape
    return field[N0 // 2, N1 // 2, :, :]

def extract_field_stats(field):
    """Compute amplitude and phase statistics from the central slice of a field."""
    central_slice = extract_central_slice(field)
    amplitude = np.abs(central_slice)
    phase = np.angle(central_slice)
    stats = {
        "amp_mean": np.mean(amplitude),
        "amp_std": np.std(amplitude),
        "phase_mean": np.mean(phase),
        "phase_std": np.std(phase)
    }
    return stats

def parallel_parameter_search(v_e_values, fixed_params):
    """
    Run DV-RIPE simulations in parallel over a list of v_e_values.
    Returns a list of diagnostics dictionaries.
    """
    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_simulation_for_v_e, v, fixed_params): v for v in v_e_values}
        for future in as_completed(futures):
            v = futures[future]
            diag = future.result()
            if diag is not None:
                results.append(diag)
                print(f"v_e = {diag['v_e']:.3f} | Effective Spin: {diag['effective_spin']:.4f} | Charge: {diag['charge']:.4f} | Energy: {diag['energy_proxy']:.4f} | Amp_mean: {diag['amp_mean']:.4e} | Phase_std: {diag['phase_std']:.4e}")
            else:
                print(f"Simulation failed for v_e = {v:.3f}")
    return results

def save_results_to_csv(results, filename="parallel_search_results.csv"):
    """Save the diagnostics results to a CSV file (kept compact)."""
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["v_e", "raw_spin", "effective_spin", "charge", "energy_proxy", "amp_mean", "amp_std", "phase_mean", "phase_std"]
        writer.writerow(header)
        for diag in results:
            writer.writerow([
                diag["v_e"], diag["raw_spin"], diag["effective_spin"],
                diag["charge"], diag["energy_proxy"], diag["amp_mean"],
                diag["amp_std"], diag["phase_mean"], diag["phase_std"]
            ])
    print("Results written to", filename)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Fixed simulation parameters: use a large, ambient domain.
    fixed_params = {
        "field_shape": (16, 32, 64, 64),    # enlarged domain
        "gauge_shape": (4, 32, 64, 64),
        "grav_shape": (64, 64, 64),
        "tau_end": 5.0,    # extended duration
        "dx": 0.1,
        "dt": 0.01,
        "lambda_e": 1.0,
        "v_e": 1.0,        # will be varied
        "delta_e": 0.1,
        "e_gauge": 0.1,
        "adaptive": False,
        "err_tolerance": 1e-3,
        "dt_min": 1e-6,
        "dt_max": 0.1
    }
    
    # Define a wide range for v_e.
    v_e_values = np.linspace(0.1, 5.0, 20)  # 20 different values
    
    results = parallel_parameter_search(v_e_values, fixed_params)
    save_results_to_csv(results, filename="parallel_search_results.csv")

if __name__ == "__main__":
    main()
