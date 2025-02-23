#!/usr/bin/env python3
"""
drivers/parameter_search_multi.py

This script performs a multi-objective optimization on the DV-RIPE simulation
to search for resonant conditions that yield a stable vortex with emergent properties:
  - Effective spin ~ 0.5,
  - Net charge = -1, and
  - An energy proxy corresponding to 511 keV (dimensionless value = 1).

We define a pymoo problem that calls the actual simulation (run_dvripe_sim)
with an enlarged domain, extended simulation duration, GPU acceleration enabled, and ambient seeding.
The objectives are:
  f1 = |effective_spin - 0.5|
  f2 = |charge + 1|
  f3 = |energy_proxy - 1|

Effective spin is computed by mapping raw spin via: effective_spin = -0.5 * raw_spin.
"""

import os
import sys
import numpy as np
import logging

# Ensure the project root is in the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import run_dvripe_sim
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define target values.
TARGET_EFFECTIVE_SPIN = 0.5   # Target effective spin (e.g., corresponding to 1/2)
TARGET_CHARGE = -1.0          # Electron charge.
TARGET_ENERGY = 1.0           # Energy proxy corresponding to 511 keV (dimensionless)

# Fixed simulation parameters.
FIXED_PARAMS = {
    "field_shape": (16, 32, 64, 64),    # Enlarged domain.
    "gauge_shape": (4, 32, 64, 64),
    "grav_shape": (64, 64, 64),
    "tau_end": 5.0,                     # Extended simulation duration.
    "dx": 0.1,
    "dt": 0.01,
    "adaptive": False,
    "dt_min": 1e-6,
    "dt_max": 0.1,
    "err_tolerance": 1e-3,
    # The four parameters to be optimized are:
    # "lambda_e": will be varied,
    # "v_e": will be varied,
    # "delta_e": will be varied,
    # "e_gauge": will be varied.
    "use_gpu": True,                    # Enable GPU acceleration.
}

class DVRIPEProblem(ElementwiseProblem):
    def __init__(self):
        # Decision variables: lambda_e, v_e, delta_e, e_gauge.
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
            # run_dvripe_sim returns (raw_spin, charge, energy_proxy).
            result = run_dvripe_sim(sim_params)
            raw_spin = result[0]
            charge = result[1]
            energy_proxy = result[2]
            # Force conversion of raw_spin to float if necessary.
            effective_spin = -0.5 * float(raw_spin)
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
    
    print("Optimization complete.")
    print("Pareto-optimal solutions:")
    for sol in res.F:
        print(sol)
    
    Scatter().add(res.F).show()
    
    output_data = {
        "X": res.X.tolist(),
        "F": res.F.tolist()
    }
    with open("best_params_multi.json", "w") as f:
        import json
        json.dump(output_data, f, indent=4)
    print("Results saved to best_params_multi.json")

if __name__ == "__main__":
    main()
