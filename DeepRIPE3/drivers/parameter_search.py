#!/usr/bin/env python3
"""
drivers/parameter_search.py

This script scans the parameter space of DV-RIPE to identify resonant conditions that yield
a stable vortex with emergent properties:
  - Effective spin ~ 0.5,
  - Net charge = -1,
  - And an energy proxy corresponding to 511 keV (dimensionless value = 1).

GPU acceleration is forced on by setting "use_gpu": True.
Extra logging has been added in the gauge operator (via GPUKernels) to verify GPU usage.
"""

import os
import sys
import json
import logging
from skopt import gp_minimize
from skopt.space import Real

# Ensure the project root is in the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import run_dvripe_sim

# --- Target diagnostics (tunable) ---
TARGET_EFFECTIVE_SPIN = 0.5   # Target effective spin (after mapping raw spin)
TARGET_CHARGE = -1.0          # Target net charge
TARGET_ENERGY_DIMLESS = 1.0   # Energy proxy corresponding to 511 keV (dimensionless)

# --- Weights for the composite objective ---
WEIGHT_SPIN = 1.0
WEIGHT_CHARGE = 1.0
WEIGHT_ENERGY = 1.0

def composite_objective(params):
    """
    Composite objective function for the parameter search.

    params: list [lambda_e, v_e, delta_e, e_gauge]

    Constructs a simulation using an enlarged domain, extended duration, and GPU acceleration enabled.
    Then extracts diagnostics (raw_spin, charge, energy_proxy) and maps raw spin to effective spin.
    Returns the weighted absolute deviation from the target values.
    """
    lambda_e, v_e, delta_e, e_gauge = params
    sim_params = {
        "field_shape": (16, 32, 64, 64),   # Enlarged domain
        "gauge_shape": (4, 32, 64, 64),
        "grav_shape": (64, 64, 64),
        "tau_end": 5.0,                    # Extended simulation duration
        "dx": 0.1,
        "dt": 0.01,
        "lambda_e": lambda_e,
        "v_e": v_e,
        "delta_e": delta_e,
        "e_gauge": e_gauge,
        "adaptive": False,
        "use_gpu": True                    # Force GPU acceleration
    }
    try:
        # run_dvripe_sim returns (raw_spin, charge, energy_proxy)
        result = run_dvripe_sim(sim_params)
        raw_spin = result[0]
        charge = result[1]
        energy_proxy = result[2]
        effective_spin = -0.5 * raw_spin  # For example, raw_spin of -1 yields effective spin 0.5.
    except Exception as e:
        logging.error(f"Simulation error with params {params}: {e}")
        return 1e6  # Large penalty if simulation fails.
    
    err_spin = abs(effective_spin - TARGET_EFFECTIVE_SPIN)
    err_charge = abs(charge - TARGET_CHARGE)
    err_energy = abs(energy_proxy - TARGET_ENERGY_DIMLESS)
    objective_value = WEIGHT_SPIN * err_spin + WEIGHT_CHARGE * err_charge + WEIGHT_ENERGY * err_energy

    logging.info(f"Params: {params} => Effective Spin: {effective_spin:.4f}, Charge: {charge:.4f}, Energy: {energy_proxy:.4f}, Obj: {objective_value:.4f}")
    return objective_value

def run_parameter_search(n_calls=50):
    """
    Run the parameter search using gp_minimize over the following dimensions:
      - lambda_e in [0.1, 5.0]
      - v_e in [0.1, 5.0]
      - delta_e in [-2.5, 2.5]
      - e_gauge in [0.001, 1.0]
    """
    dimensions = [
        Real(0.1, 5.0, name="lambda_e"),
        Real(0.1, 5.0, name="v_e"),
        Real(-2.5, 2.5, name="delta_e"),
        Real(0.001, 1.0, name="e_gauge")
    ]
    result = gp_minimize(composite_objective, dimensions, n_calls=n_calls, random_state=42)
    return result.x, result.fun

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    best_params, best_obj = run_parameter_search(n_calls=50)
    print("Best parameters found:", best_params)
    print("Objective value:", best_obj)
    output_data = {"best_params": best_params, "objective_value": best_obj}
    with open("best_params_composite.json", "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info("Best parameters written to best_params_composite.json")

if __name__ == "__main__":
    main()
