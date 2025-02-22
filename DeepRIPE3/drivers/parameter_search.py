#!/usr/bin/env python3
"""
drivers/parameter_search.py

This script scans the parameter space of DV-RIPE to find resonant storm conditions
that yield a stable vortex with emergent properties:
  - Effective spin ~ 0.5 (from a composite of two interacting fields)
  - Net charge = -1
  - An energy proxy (mass) corresponding to 511 keV (dimensionless value = 1)

It uses skopt's gp_minimize to minimize a composite objective function that runs a
short simulation and compares the diagnostics to the target values.
"""

import os
import sys
import json
import logging
from skopt import gp_minimize
from skopt.space import Real

# Ensure the project root is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import run_dvripe_sim
from src.constants import energy_dimless_to_keV  # if needed later

# --- Target Diagnostics (tunable) ---
TARGET_EFFECTIVE_SPIN = 0.5   # Our target effective spin (composite result)
TARGET_CHARGE = -1.0          # Net charge target
TARGET_ENERGY_DIMLESS = 1.0   # Dimensionless energy corresponding to 511 keV

# --- Weights for the composite objective ---
WEIGHT_SPIN = 1.0
WEIGHT_CHARGE = 1.0
WEIGHT_ENERGY = 1.0

def composite_objective(params):
    """
    Composite objective function for the parameter search.
    
    params: list [lambda_e, v_e, delta_e, e_gauge]
    
    Steps:
      1. Construct a short-run simulation using these parameters.
      2. Extract diagnostics: raw_spin, charge, grav.
      3. Map raw_spin into effective_spin via: effective_spin = -0.5 * raw_spin.
         (So, for example, a raw winding of -1 gives an effective spin of 0.5.)
      4. Compute the weighted absolute errors against target effective spin, charge, and energy.
    """
    lambda_e, v_e, delta_e, e_gauge = params
    sim_params = {
        "field_shape": (4, 8, 16, 16),
        "gauge_shape": (4, 8, 16, 16),
        "grav_shape": (16, 16, 16),
        "tau_end": 0.5,   # Short simulation duration for scanning
        "dx": 0.1,
        "dt": 0.01,
        "lambda_e": lambda_e,
        "v_e": v_e,
        "delta_e": delta_e,
        "e_gauge": e_gauge,
        "adaptive": False,
    }
    try:
        result = run_dvripe_sim(sim_params)
        # Use only the first three returned values:
        raw_spin = result[0]
        charge = result[1]
        grav = result[2]
        effective_spin = -0.5 * raw_spin  # Mapping: raw -1 gives effective 0.5.
        effective_energy = grav           # Placeholder for energy diagnostics.
    except Exception as e:
        logging.error("Simulation error with params {}: {}".format(params, e))
        return 1e6  # Large penalty if simulation fails.

    err_spin = abs(effective_spin - TARGET_EFFECTIVE_SPIN)
    err_charge = abs(charge - TARGET_CHARGE)
    err_energy = abs(effective_energy - TARGET_ENERGY_DIMLESS)
    objective_value = WEIGHT_SPIN * err_spin + WEIGHT_CHARGE * err_charge + WEIGHT_ENERGY * err_energy

    logging.info("Params: {} => Effective Spin: {:.4f}, Charge: {:.4f}, Energy: {:.4f}, Obj: {:.4f}"
                 .format(params, effective_spin, charge, effective_energy, objective_value))
    return objective_value

def run_parameter_search(n_calls=50):
    """
    Run the parameter search over:
      - lambda_e in [0.5, 2.0]
      - v_e in [0.5, 2.0]
      - delta_e in [-0.5, 0.5]
      - e_gauge in [0.01, 0.5]
    """
    dimensions = [
        Real(0.5, 2.0, name="lambda_e"),
        Real(0.5, 2.0, name="v_e"),
        Real(-0.5, 0.5, name="delta_e"),
        Real(0.01, 0.5, name="e_gauge")
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
