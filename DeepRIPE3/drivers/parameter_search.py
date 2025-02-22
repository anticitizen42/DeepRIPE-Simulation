#!/usr/bin/env python3
"""
Adaptive Parameter Search for DV/RIPE Simulation Framework with Physical Diagnostics

This script uses skopt's gp_minimize to search over the parameter space.
The objective function runs a short simulation (via run_dvripe_sim) and compares
the resulting diagnostics (spin, charge, gravitational indentation) to target values.
The objective is the weighted sum of deviations.
"""

import sys
import os
import logging
import json
from skopt import gp_minimize
from skopt.space import Real

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import run_dvripe_sim

# ------------------------------------------------------------------
# TARGET DIAGNOSTICS & WEIGHTS (adjust as needed)
# ------------------------------------------------------------------
TARGET_SPIN = 0.5
TARGET_CHARGE = 0.0
TARGET_MASS = 0.8  # using gravitational indentation as a mass proxy

WEIGHT_SPIN = 1.0
WEIGHT_CHARGE = 1.0
WEIGHT_MASS = 1.0

# ------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ------------------------------------------------------------------
def objective(params):
    """
    The params vector is assumed to be: [lambda_e, v_e, delta_e, e_gauge].
    
    We run the simulation for a short time (tau_end=0.5) to obtain the physical diagnostics,
    then compute the weighted error compared to the target diagnostics.
    """
    lambda_e, v_e, delta_e, e_gauge = params
    # Construct simulation parameters for a short run (speed is key in optimization)
    sim_params = {
        "field_shape": (4, 8, 8, 8),
        "gauge_shape": (4, 8, 8, 8),
        "grav_shape": (8, 8, 8),
        "tau_end": 0.5,   # short simulation duration for parameter search
        "dx": 0.1,
        "dt": 0.01,
        "adaptive": False,  # use fixed-step for consistency
        "lambda_e": lambda_e,
        "v_e": v_e,
        "delta_e": delta_e,
        "e_gauge": e_gauge,
    }
    try:
        spin, charge, grav = run_dvripe_sim(sim_params)
    except Exception as e:
        logging.error(f"Simulation error with params {params}: {e}")
        # Return a high penalty if the simulation fails
        return 1e6
    
    # Compute absolute deviations from target diagnostics
    err_spin   = abs(spin - TARGET_SPIN)
    err_charge = abs(charge - TARGET_CHARGE)
    err_mass   = abs(grav - TARGET_MASS)
    
    obj_value = WEIGHT_SPIN * err_spin + WEIGHT_CHARGE * err_charge + WEIGHT_MASS * err_mass
    logging.info(f"Params: {params} => Spin: {spin:.3f}, Charge: {charge:.3f}, Mass: {grav:.3f}, Obj: {obj_value:.3f}")
    return obj_value

# ------------------------------------------------------------------
# RUN PARAMETER SEARCH
# ------------------------------------------------------------------
def run_physical_parameter_search(initial_bounds, n_calls=50):
    dimensions = [
        Real(initial_bounds[0][0], initial_bounds[0][1], name="lambda_e"),
        Real(initial_bounds[1][0], initial_bounds[1][1], name="v_e"),
        Real(initial_bounds[2][0], initial_bounds[2][1], name="delta_e"),
        Real(initial_bounds[3][0], initial_bounds[3][1], name="e_gauge")
    ]
    
    res = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)
    return res.x, res.fun

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Define initial search bounds for parameters: [lambda_e, v_e, delta_e, e_gauge]
    initial_bounds = [
        (0.5, 2.0),    # lambda_e
        (0.5, 2.0),    # v_e
        (-0.5, 0.5),   # delta_e
        (0.01, 0.5)    # e_gauge
    ]
    
    best_params, best_obj = run_physical_parameter_search(initial_bounds, n_calls=50)
    print("Best parameters found:", best_params)
    print("Objective value:", best_obj)
    
    # Save the best parameters to a JSON file for use in simulation
    output_data = {
        "best_params": best_params,
        "objective_value": best_obj,
    }
    output_file = "best_params_physical.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"Best parameters written to {output_file}")

if __name__ == "__main__":
    main()
