#!/usr/bin/env python3
"""
drivers/simulation_runner.py

A simple simulation runner for the DV‑RIPE model.

This script:
  1. Loads a default (or candidate) parameter set.
  2. Initializes the electron, gauge, and gravity fields.
  3. Evolves the system using the implicit integrator from tau = 0 to tau = tau_end.
  4. Prints out heartbeat messages, the integration log (diagnostics at each step), and final diagnostics:
       - Effective spin,
       - Net charge, and
       - Gravitational indentation (energy proxy).

This script does not perform any parameter scanning—it simply runs the simulation using the provided parameters.
"""

import os
import json
import logging
import numpy as np

from src.simulation import run_dvripe_sim

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_parameters(filename="default_simulation_params.json"):
    """
    Load simulation parameters from a JSON file.
    If the file does not exist, return a default parameter set.
    """
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                params = json.load(f)
            logging.info("Loaded parameters from %s", filename)
            return params
        except Exception as e:
            logging.error("Error loading parameters from %s: %s", filename, e)
    # Default parameter set for DV‑RIPE simulation.
    params = {
        "field_shape": [16, 32, 64, 64],    # For polar mode, only first two dims are used for electron grid.
        "grav_shape": [64, 64, 64],
        "tau_end": 5.0,
        "dx": 0.1,
        "dt": 0.01,
        "integrator": "implicit",
        "use_gpu": True,
        "polar": True,
        "radial_points": 64,
        "angular_points": 64,
        "r_max": 1.0,
        "lambda_e": 1.0,
        "v_e": 1.0,
        "delta_e": 0.1,
        "e_gauge": 0.1,
        "gamma": 0.0,
        "eta": 1.0,
        "Lambda": 0.5,
        "kappa": 1.0,
        "G_eff": 1.0,
        "mu": 0.1,
        "dt_min": 1e-8,
        "verbose_integration": True,
        "gauge_coupling": 1.0
    }
    # For polar mode, define gauge_shape as (3, 4, N0, N1, radial_points, angular_points)
    N0 = params["field_shape"][0]
    N1 = params["field_shape"][1]
    params["gauge_shape"] = [3, 4, N0, N1, params["radial_points"], params["angular_points"]]
    logging.info("Using default simulation parameters.")
    return params

def print_integration_log(log):
    """
    Print the integration log entries.
    """
    print("Integration Log:")
    for entry in log:
        print(f"  tau = {entry.get('tau', 0):.8f}, spin = {entry.get('spin', 0):.4f}, "
              f"charge = {entry.get('charge', 0):.4f}, energy = {entry.get('energy', 0):.4f}")

def main():
    params = load_parameters()
    
    logging.info("Simulation parameters: %s", params)
    logging.info("Running simulation for tau_end = %s", params.get("tau_end"))
    
    try:
        result = run_dvripe_sim(params)
    except Exception as e:
        logging.error("Simulation failed: %s", e)
        return
    
    # Check if integration log was returned.
    if isinstance(result, tuple) and len(result) == 4:
        spin, charge, energy, integration_log = result
    else:
        spin, charge, energy = result
        integration_log = None
    
    print("Simulation complete.")
    print("Final Diagnostics:")
    print(f"  Spin: {spin:.4f}")
    print(f"  Charge: {charge:.4f}")
    print(f"  Energy Proxy: {energy:.4f}")
    
    if integration_log is not None:
        print_integration_log(integration_log)
    
if __name__ == "__main__":
    main()
