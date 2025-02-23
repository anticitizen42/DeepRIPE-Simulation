#!/usr/bin/env python3
"""
drivers/troubleshoot_simulation.py

This script runs a series of DV-RIPE simulations while varying one parameter (v_e)
over a wide range. For each simulation it:
  - Uses an extended simulation duration (tau_end = 5.0) and an enlarged domain.
  - Employs ambient initial conditions so that small random fluctuations may generate transient vortices.
  - Extracts detailed diagnostics from the final electron field φ₁:
      • Amplitude and phase statistics from the central 2D slice.
      • The topological winding (raw spin) computed from that slice.
      • The net charge and energy proxy.
  - Writes a summarized CSV file (kept under ~20 KB) for offline analysis.
"""

import numpy as np
import csv
import os

# Import simulation and diagnostic functions
from src.simulation import run_dvripe_sim
from src.diagnostics import compute_spin, compute_charge, compute_gravity_indentation

def extract_central_slice(field):
    """
    Given a 4D field (N0, N1, Ny, Nz), extract the central 2D slice.
    """
    N0, N1, Ny, Nz = field.shape
    return field[N0 // 2, N1 // 2, :, :]

def get_field_stats(phi_slice):
    """
    Compute basic statistics (mean, std, min, max) for amplitude and phase from a 2D complex field.
    """
    amplitude = np.abs(phi_slice)
    phase = np.angle(phi_slice)
    stats = {
        "amp_mean": np.mean(amplitude),
        "amp_std": np.std(amplitude),
        "amp_min": np.min(amplitude),
        "amp_max": np.max(amplitude),
        "phase_mean": np.mean(phase),
        "phase_std": np.std(phase),
        "phase_min": np.min(phase),
        "phase_max": np.max(phase)
    }
    return stats

def run_simulation_for_params(v_e_value, fixed_params):
    """
    Run the simulation with a given v_e value (other parameters fixed) and return diagnostics.
    """
    sim_params = fixed_params.copy()
    sim_params["v_e"] = v_e_value
    sim_params["tau_end"] = 5.0  # extended duration

    # Run simulation; run_dvripe_sim returns (spin, charge, grav).
    # Spin is now computed from the final electron field (phi1).
    try:
        result = run_dvripe_sim(sim_params)
        raw_spin = result[0]    # computed from phi1 via winding diagnostic
        charge = result[1]
        energy_proxy = result[2]
    except Exception as e:
        print("Simulation error for v_e = {}: {}".format(v_e_value, e))
        return None

    # For additional diagnostics, we want to inspect the final electron field.
    # If possible, we try to re-run or extract phi1_fin.
    try:
        # Assume run_dvripe_sim now returns 4 values (spin, charge, grav, phi1_fin).
        phi1_fin = run_dvripe_sim(sim_params)[3]
    except IndexError:
        # If not available, we cannot compute extra stats.
        phi1_fin = None

    diagnostics = {
        "v_e": v_e_value,
        "raw_spin": raw_spin,
        "effective_spin": -0.5 * raw_spin,  # our mapping: raw -1 -> effective 0.5
        "charge": charge,
        "energy_proxy": energy_proxy,
    }

    if phi1_fin is not None:
        central_slice = extract_central_slice(phi1_fin)
        stats = get_field_stats(central_slice)
        diagnostics.update(stats)
    else:
        # If phi1_fin is unavailable, we fill in zeros.
        diagnostics.update({
            "amp_mean": 0.0,
            "amp_std": 0.0,
            "amp_min": 0.0,
            "amp_max": 0.0,
            "phase_mean": 0.0,
            "phase_std": 0.0,
            "phase_min": 0.0,
            "phase_max": 0.0
        })
    return diagnostics

def main():
    # Fixed simulation parameters: using our enlarged domain and ambient seeding.
    fixed_params = {
        "field_shape": (16, 32, 64, 64),    # quadrupled dimensions
        "gauge_shape": (4, 32, 64, 64),
        "grav_shape": (64, 64, 64),
        "tau_end": 5.0,    # extended simulation duration
        "dx": 0.1,
        "dt": 0.01,
        "lambda_e": 1.0,
        "v_e": 1.0,        # this will be varied
        "delta_e": 0.1,
        "e_gauge": 0.1,
        "adaptive": False,
        "err_tolerance": 1e-3,
        "dt_min": 1e-6,
        "dt_max": 0.1
    }
    
    # Vary v_e over a wide range.
    v_e_values = np.linspace(0.1, 5.0, 20)  # 20 values
    results = []
    for v_e in v_e_values:
        diag = run_simulation_for_params(v_e, fixed_params)
        if diag is not None:
            results.append(diag)
            print("v_e = {:.3f} | Effective Spin: {:.4f} | Charge: {:.4f} | Energy: {:.4f} | Amp_mean: {:.4e} | Phase_std: {:.4e}"
                  .format(diag["v_e"], diag["effective_spin"], diag["charge"], diag["energy_proxy"],
                          diag["amp_mean"], diag["phase_std"]))
    
    # Write results to CSV (limit rows to ensure file stays under ~20 KB).
    csv_filename = "troubleshoot_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["v_e", "raw_spin", "effective_spin", "charge", "energy_proxy",
                  "amp_mean", "amp_std", "amp_min", "amp_max",
                  "phase_mean", "phase_std", "phase_min", "phase_max"]
        writer.writerow(header)
        for diag in results:
            writer.writerow([
                diag["v_e"], diag["raw_spin"], diag["effective_spin"],
                diag["charge"], diag["energy_proxy"], diag["amp_mean"],
                diag["amp_std"], diag["amp_min"], diag["amp_max"],
                diag["phase_mean"], diag["phase_std"], diag["phase_min"], diag["phase_max"]
            ])
    print("Troubleshooting CSV written to", csv_filename)

if __name__ == "__main__":
    main()
