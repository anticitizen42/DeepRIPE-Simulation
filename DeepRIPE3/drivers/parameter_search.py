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
During optimization, candidate solutions (their parameters and objective errors) are stored in a database.
If you press Ctrl+C, the optimization stops gracefully and the current database is printed and saved.
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
TARGET_EFFECTIVE_SPIN = 0.5   # Desired effective spin (half-integer)
TARGET_CHARGE = -1.0          # Desired net charge
# For energy, the target is set equal to mass_scale (1.0 for electron, 1836 for proton)
TARGET_ENERGY = None  # Will be set equal to mass_scale below.

# --- Tolerance for acceptable solutions ---
TOLERANCE = 1e-3

# --- Fixed simulation parameters ---
FIXED_PARAMS = {
    "field_shape": (16, 32, 64, 64),
    "gauge_shape": (3, 4, 32, 64, 64),
    "grav_shape": (64, 64, 64),
    "tau_end": 5.0,
    "dx": 0.1,
    "dt": 0.01,
    "adaptive": False,
    "dt_min": 1e-6,
    "dt_max": 0.1,
    "err_tolerance": 1e-3,
    # Optimized parameters: lambda_e, v_e, delta_e, e_gauge.
    "use_gpu": True,
    "mass_scale": 1.0,  # Set to 1.0 for electron; change to 1836 for proton.
}
TARGET_ENERGY = FIXED_PARAMS["mass_scale"]

class DVRIPEProblem(ElementwiseProblem):
    def __init__(self):
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
            effective_spin = -0.5 * raw_spin  # e.g. raw_spin = -1 yields effective_spin = 0.5.
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
    database = []
    res = None
    try:
        problem = DVRIPEProblem()
        algorithm = NSGA2(pop_size=40)
        
        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', 20),
                       seed=42,
                       save_history=True,
                       verbose=True)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Terminating optimization early.")
        # Attempt to retrieve the last available optimal population.
        try:
            if hasattr(algorithm, "history") and algorithm.history:
                res = algorithm.history[-1].opt
            else:
                print("No history available; using current results if any.")
        except Exception as e:
            print("Error retrieving last generation from history:", e)
    finally:
        if res is not None:
            for x, f in zip(res.X, res.F):
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
        else:
            print("No results were obtained.")
        
        print("\nCandidate Solutions Database:")
        for cand in database:
            print(cand)
        
        output_filename = "electron_solution_database.json"
        with open(output_filename, "w") as f:
            json.dump(database, f, indent=4)
        print(f"\nFull database saved to {output_filename}")

if __name__ == "__main__":
    main()
