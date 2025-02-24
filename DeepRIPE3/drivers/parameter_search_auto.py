#!/usr/bin/env python3
"""
drivers/parameter_search_auto.py

This script performs a multi-objective optimization on the DV-RIPE simulation to search for
parameter sets that yield the desired electron-like (or proton-like) observables:
  - Effective spin ~ 0.5  (computed via effective_spin = -0.5 * raw_spin),
  - Net charge = -1,
  - Energy proxy = mass_scale (1.0 for electron, 1836 for proton).

The simulation is run with an enlarged domain, extended duration, GPU acceleration enabled,
and ambient seeding. We use the NSGA-II algorithm from pymoo to find a Pareto front of solutions.
After optimization, the script builds a database (in JSON format) of candidate solutions
with their optimized parameters and corresponding diagnostic errors.
"""

import os
import sys
import json
import logging
import numpy as np

# Ensure the project root is in the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from src.simulation import run_dvripe_sim

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Target Observables ---
TARGET_EFFECTIVE_SPIN = 0.5   # Target effective spin (half-integer)
TARGET_CHARGE = -1.0          # Target net charge
# For the energy proxy, the target is set to mass_scale:
# For an electron, mass_scale=1.0 (TARGET_ENERGY = 1.0).
# For a proton, mass_scale=1836 (TARGET_ENERGY = 1836).
# Here, we set up the target energy to be equal to the mass_scale.
# (This value must match the mass_scale provided in FIXED_PARAMS.)
TARGET_ENERGY = None  # will be set equal to mass_scale below

# --- Tolerance for acceptable solutions (per objective) ---
TOLERANCE = 1e-3

# --- Fixed simulation parameters ---
# Change "mass_scale" here to 1.0 for electrons or 1836 for protons.
FIXED_PARAMS = {
    "field_shape": (16, 32, 64, 64),    # Enlarged domain for electron fields.
    "gauge_shape": (4, 32, 64, 64),      # Enlarged domain for gauge field.
    "grav_shape": (64, 64, 64),          # Enlarged gravity field.
    "tau_end": 5.0,                     # Extended simulation duration.
    "dx": 0.1,
    "dt": 0.01,
    "adaptive": False,
    "dt_min": 1e-6,
    "dt_max": 0.1,
    "err_tolerance": 1e-3,
    # The following four parameters will be optimized:
    # "lambda_e", "v_e", "delta_e", "e_gauge"
    "use_gpu": True,                    # Enable GPU acceleration.
    "mass_scale": 1.0,                  # Set to 1.0 for electron; change to 1836 for proton.
}

# Set target energy equal to the mass_scale in FIXED_PARAMS.
TARGET_ENERGY = FIXED_PARAMS["mass_scale"]

class DVRIPEProblem(ElementwiseProblem):
    def __init__(self):
        # Optimize 4 parameters: lambda_e, v_e, delta_e, e_gauge.
        super().__init__(n_var=4,
                         n_obj=3,
                         xl=np.array([0.1, 0.1, -2.5, 0.001]),
                         xu=np.array([5.0, 5.0, 2.5, 1.0]))
    
    def _evaluate(self, x, out, *args, **kwargs):
        lambda_e, v_e, delta_e, e_gauge = x
        sim_params = FIXED_PARAMS.copy()
        sim_params["lambda_e"] = lambda_e
        sim_params["v_e"] = v_e
        sim_params["delta_e"] = delta_e
        sim_params["e_gauge"] = e_gauge
        
        try:
            # run_dvripe_sim returns (raw_spin, charge, energy_proxy)
            result = run_dvripe_sim(sim_params)
            raw_spin = result[0]
            charge = result[1]
            energy_proxy = result[2]
            effective_spin = -0.5 * raw_spin  # Mapping: e.g. raw_spin = -1 gives effective_spin = 0.5.
        except Exception as e:
            logging.error(f"Simulation error with params {x}: {e}")
            out["F"] = [1e6, 1e6, 1e6]
            return
        
        f1 = abs(effective_spin - TARGET_EFFECTIVE_SPIN)
        f2 = abs(charge - TARGET_CHARGE)
        f3 = abs(energy_proxy - TARGET_ENERGY)
        out["F"] = [f1, f2, f3]
        logging.info(f"Params: {x} => Effective Spin: {effective_spin:.4f}, Charge: {charge:.4f}, Energy: {energy_proxy:.4f}, Objectives: {[f1, f2, f3]}")

def main():
    problem = DVRIPEProblem()
    algorithm = NSGA2(pop_size=40)
    
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 20),
                   seed=42,
                   save_history=True,
                   verbose=True)
    
    # Build a database of candidate solutions.
    database = []
    for x, f in zip(res.X, res.F):
        # Compute effective spin from simulation (from our simulation logs, effective_spin is -0.5*raw_spin).
        # Here we simply record the parameters and the objective errors.
        candidate = {
            "params": {
                "lambda_e": float(x[0]),
                "v_e": float(x[1]),
                "delta_e": float(x[2]),
                "e_gauge": float(x[3])
            },
            "objectives": {
                "spin_error": float(f[0]),
                "charge_error": float(f[1]),
                "energy_error": float(f[2])
            }
        }
        database.append(candidate)
    
    # Save the full database to a JSON file.
    output_filename = "electron_solution_database.json"
    with open(output_filename, "w") as f:
        json.dump(database, f, indent=4)
    
    # Filter candidates that meet target observables within tolerance.
    acceptable_candidates = [cand for cand in database
                             if abs(cand["objectives"]["spin_error"]) < TOLERANCE
                             and abs(cand["objectives"]["charge_error"]) < TOLERANCE
                             and abs(cand["objectives"]["energy_error"]) < TOLERANCE]
    
    print("Optimization complete.")
    print(f"Found {len(acceptable_candidates)} candidate solutions meeting the target observables:")
    for cand in acceptable_candidates:
        print(cand)
    
    print("\nPareto-optimal objective values:")
    for sol in res.F:
        print(sol)
    
    Scatter().add(res.F).show()
    print(f"Full database of candidate solutions saved to {output_filename}")

if __name__ == "__main__":
    main()
