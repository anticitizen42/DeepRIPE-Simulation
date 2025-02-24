#!/usr/bin/env python3
"""
drivers/parameter_search_multi.py

This script performs a multi-objective optimization on the DV-RIPE simulation
to search for resonant conditions that yield an electron vortex with:
  - Effective spin ~ 0.5,
  - Net charge = -1, and
  - Energy proxy = 1.

The candidate parameter vector now has ten components:
  [λₑ, vₑ, δₑ, e_gauge, γ, η, Λ, κ, G_eff, μ]
where:
  - λₑ, vₑ, δₑ, and e_gauge control the scalar potential and gauge coupling,
  - γ is a damping parameter (added as –γ·φ in the PDE operator),
  - η scales the nonlinear potential term,
  - Λ (Lambda) is the Collapse Metric,
  - κ (kappa) is the membrane coupling/tension parameter,
  - G_eff scales the gravitational coupling,
  - μ (mu) is a diffusivity/viscosity parameter.

This version automatically refines the parameter bounds (zoom iterations) and, if the flag "verbose_integration"
is enabled, prints the intermediate integration diagnostics (spin, charge, and energy after each step).
"""

import os
import sys
import numpy as np
import json
import logging
from multiprocessing import Pool, cpu_count

# Ensure project root is in the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation import run_dvripe_sim
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Target diagnostics.
TARGET_SPIN   = 0.5
TARGET_CHARGE = -1.0
TARGET_ENERGY = 1.0

# Tolerances for candidate acceptance.
TOL_SPIN   = 0.05
TOL_CHARGE = 0.05
TOL_ENERGY = 0.05

# Fixed simulation parameters.
FIXED_PARAMS = {
    "field_shape": (16, 32, 64, 64),    # For polar mode, only the first two dims are used for electron initialization.
    "grav_shape": (64, 64, 64),
    "tau_end": 5.0,
    "dx": 0.1,
    "dt": 0.01,
    "integrator": "implicit",
    "use_gpu": True,
    "polar": True,
    "radial_points": 64,
    "angular_points": 64,
    "r_max": 1.0,
    "gamma": 0.0,    # Damping parameter.
    "eta": 1.0,      # Nonlinear scaling factor.
    "Lambda": 0.5,   # Collapse Metric.
    "kappa": 1.0,    # Membrane coupling.
    "G_eff": 1.0,    # Gravitational scaling.
    "mu": 0.1,       # Diffusivity/viscosity.
    "verbose_integration": True  # Flag to return integration log.
}
# For polar mode, gauge_shape is defined as (3, 4, N0, N1, radial_points, angular_points).
N0 = FIXED_PARAMS["field_shape"][0]
N1 = FIXED_PARAMS["field_shape"][1]
FIXED_PARAMS["gauge_shape"] = (3, 4, N0, N1,
                               FIXED_PARAMS.get("radial_points", 64),
                               FIXED_PARAMS.get("angular_points", 64))

DATABASE_FILENAME = "electron_solution_database.json"

# Initial candidate bounds for ten variables:
# [λₑ, vₑ, δₑ, e_gauge, γ, η, Λ, κ, G_eff, μ]
INITIAL_XL = np.array([0.1, 0.1, -2.5, 0.001, -1.0, 0.5, 0.0, 0.001, 0.1, 0.001])
INITIAL_XU = np.array([5.0, 5.0,  2.5, 1.0,  1.0, 2.0, 1.0, 10.0, 10.0, 1.0])

# Zooming parameters.
NUM_ZOOM_ITER = 3
BEST_FRAC     = 0.2
MARGIN_FACTOR = 0.10

class DVRIPEProblem(ElementwiseProblem):
    def __init__(self, xl, xu):
        super().__init__(n_var=10, n_obj=3, xl=xl, xu=xu)
    
    def _evaluate(self, x, out, *args, **kwargs):
        (lambda_e, v_e, delta_e, e_gauge, gamma, eta, Lambda, kappa, G_eff, mu) = x
        sim_params = FIXED_PARAMS.copy()
        sim_params["lambda_e"] = lambda_e
        sim_params["v_e"] = v_e
        sim_params["delta_e"] = delta_e
        sim_params["e_gauge"] = e_gauge
        sim_params["gamma"] = gamma
        sim_params["eta"] = eta
        sim_params["Lambda"] = Lambda
        sim_params["kappa"] = kappa
        sim_params["G_eff"] = G_eff
        sim_params["mu"] = mu
        try:
            # If verbose_integration is True, run_dvripe_sim returns (spin, charge, energy, integration_log)
            result = run_dvripe_sim(sim_params)
            if sim_params.get("verbose_integration", False) and isinstance(result, tuple) and len(result) == 4:
                spin, charge, energy, log = result
            else:
                spin, charge, energy = result
        except Exception as e:
            logging.error(f"Simulation error with params {x}: {e}")
            out["F"] = [1e6, 1e6, 1e6]
            return
        f1 = abs(spin - TARGET_SPIN)
        f2 = abs(charge - TARGET_CHARGE)
        f3 = abs(energy - TARGET_ENERGY)
        out["F"] = [f1, f2, f3]
        logging.info(f"Candidate {x}: Spin={spin:.4f}, Charge={charge:.4f}, Energy={energy:.4f} -> Objectives: {[f1, f2, f3]}")
        # If an integration log was returned, print each step.
        if sim_params.get("verbose_integration", False) and 'log' in locals():
            logging.info("Integration log for candidate %s:", x)
            for step in log:
                logging.info("   tau=%.4f, spin=%.4f, charge=%.4f, energy=%.4f", 
                             step.get("tau", 0), step.get("spin", 0), step.get("charge", 0), step.get("energy", 0))
        
def evaluate_candidate(x):
    (lambda_e, v_e, delta_e, e_gauge, gamma, eta, Lambda, kappa, G_eff, mu) = x
    sim_params = FIXED_PARAMS.copy()
    sim_params["lambda_e"] = lambda_e
    sim_params["v_e"] = v_e
    sim_params["delta_e"] = delta_e
    sim_params["e_gauge"] = e_gauge
    sim_params["gamma"] = gamma
    sim_params["eta"] = eta
    sim_params["Lambda"] = Lambda
    sim_params["kappa"] = kappa
    sim_params["G_eff"] = G_eff
    sim_params["mu"] = mu
    try:
        result = run_dvripe_sim(sim_params)
        if sim_params.get("verbose_integration", False) and isinstance(result, tuple) and len(result) == 4:
            spin, charge, energy, log = result
            logging.info(f"Candidate {x} produced diagnostics: Spin={spin:.4f}, Charge={charge:.4f}, Energy={energy:.4f}")
            logging.info("Integration log for candidate %s:", x)
            for step in log:
                logging.info("   tau=%.4f, spin=%.4f, charge=%.4f, energy=%.4f", 
                             step.get("tau", 0), step.get("spin", 0), step.get("charge", 0), step.get("energy", 0))
        else:
            spin, charge, energy = result
            logging.info(f"Candidate {x} produced diagnostics: Spin={spin:.4f}, Charge={charge:.4f}, Energy={energy:.4f}")
    except Exception as e:
        logging.error(f"Error evaluating candidate {x}: {e}")
        raise e
    f1 = abs(spin - TARGET_SPIN)
    f2 = abs(charge - TARGET_CHARGE)
    f3 = abs(energy - TARGET_ENERGY)
    total_error = f1 + f2 + f3
    return (x, [f1, f2, f3], (spin, charge, energy), total_error)

def write_best_parameters(candidates):
    with open(DATABASE_FILENAME, "w") as f:
        json.dump(candidates, f, indent=4)
    logging.info(f"Best candidate parameters written to {DATABASE_FILENAME}")

def compute_new_bounds(best_params):
    best_params = np.array(best_params)  # shape (n_candidates, 10)
    new_xl = best_params.min(axis=0)
    new_xu = best_params.max(axis=0)
    margin = (new_xu - new_xl) * MARGIN_FACTOR
    new_xl = new_xl - margin
    new_xu = new_xu + margin
    new_xl = np.maximum(new_xl, INITIAL_XL)
    new_xu = np.minimum(new_xu, INITIAL_XU)
    logging.info("New lower bounds: %s", new_xl)
    logging.info("New upper bounds: %s", new_xu)
    return new_xl, new_xu

def main():
    xl, xu = INITIAL_XL.copy(), INITIAL_XU.copy()
    for zoom in range(NUM_ZOOM_ITER):
        logging.info("Zoom iteration %d/%d with bounds: %s to %s", zoom+1, NUM_ZOOM_ITER, xl, xu)
        problem = DVRIPEProblem(xl, xu)
        algorithm = NSGA2(pop_size=40)
        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', 20),
                       seed=42,
                       save_history=True,
                       verbose=True)
        logging.info("Optimization complete for zoom iteration %d.", zoom+1)
        Scatter().add(res.F).show()
        with open(f"best_params_multi_zoom_{zoom+1}.json", "w") as f:
            json.dump({"X": res.X.tolist(), "F": res.F.tolist()}, f, indent=4)
        logging.info("Raw optimization results saved for zoom iteration %d.", zoom+1)
        
        candidate_list = res.X.tolist()
        pool_size = 2 if FIXED_PARAMS["use_gpu"] else cpu_count()
        logging.info("Evaluating %d candidates in parallel using %d processes.", len(candidate_list), pool_size)
        successful_candidates = []
        with Pool(processes=pool_size) as pool:
            results = pool.map(evaluate_candidate, candidate_list)
        for x, f, diag, tot_err in results:
            if f[0] <= TOL_SPIN and f[1] <= TOL_CHARGE and f[2] <= TOL_ENERGY:
                candidate = {
                    "params": {
                        "lambda_e": float(x[0]),
                        "v_e": float(x[1]),
                        "delta_e": float(x[2]),
                        "e_gauge": float(x[3]),
                        "gamma": float(x[4]),
                        "eta": float(x[5]),
                        "Lambda": float(x[6]),
                        "kappa": float(x[7]),
                        "G_eff": float(x[8]),
                        "mu": float(x[9])
                    },
                    "objectives": {
                        "spin_error": float(f[0]),
                        "charge_error": float(f[1]),
                        "energy_error": float(f[2])
                    },
                    "diagnostics": {
                        "spin": float(diag[0]),
                        "charge": float(diag[1]),
                        "energy": float(diag[2])
                    },
                    "total_error": float(tot_err)
                }
                successful_candidates.append(candidate)
        if successful_candidates:
            write_best_parameters(successful_candidates)
            successful_candidates.sort(key=lambda cand: cand["total_error"])
            num_best = max(1, int(BEST_FRAC * len(successful_candidates)))
            best_params = [list(cand["params"].values()) for cand in successful_candidates[:num_best]]
            xl, xu = compute_new_bounds(best_params)
        else:
            logging.info("No candidates met the target criteria in zoom iteration %d.", zoom+1)
            break

if __name__ == "__main__":
    main()
