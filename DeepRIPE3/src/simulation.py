#!/usr/bin/env python3
"""
simulation.py

Runs the DV/RIPE simulation using the highest-fidelity DV-RIPE PDE solver.
This version:
  - Reads optimized simulation parameters from best_params.json (if available)
  - Initializes electron, gauge, and gravity fields
  - Chooses between fixed-step or adaptive-step evolution
  - Computes diagnostics: spin, charge, and gravitational indentation
"""

import os
import json
import numpy as np

# Field initializers
from src.fields import init_electron_fields, init_gauge_field, init_gravity_field

# PDE solvers (fixed + adaptive) and gauge utilities
from src.pde_solver import (
    evolve_fields,          # High-fidelity fixed-step solver (e.g., with fourth-order methods)
    evolve_fields_adaptive, # High-fidelity adaptive-step solver
    field_strength
)

# Diagnostics
from src.diagnostics import (
    compute_spin,
    compute_charge,
    compute_gravity_indentation
)

def load_optimized_parameters(param_file='best_params.json'):
    """
    Loads optimized parameters from a JSON file.
    Expected format: { "best_params": [lambda_e, v_e, delta_e, e_gauge], ... }
    """
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            data = json.load(f)
        return data.get("best_params", None)
    return None

def run_dvripe_sim(params):
    """
    Main DV/RIPE simulation function.

    :param params: dict with keys:
       "field_shape", "gauge_shape", "grav_shape" (tuples)
       "tau_end", "dx" (floats)
       "dt" (float, optional) - if missing, defaults to 0.01
       "lambda_e", "v_e", "delta_e", "e_gauge" (floats)
       "adaptive" (bool) - whether to use adaptive time stepping
       "dt_min", "dt_max", "err_tolerance" (for adaptive stepping)
    :return: (spin_val, charge_val, grav_val)
    """

    # Attempt to load optimized parameters from file.
    optimized_params = load_optimized_parameters()
    if optimized_params is not None:
        # Here we assume the best_params ordering is: [lambda_e, v_e, delta_e, e_gauge]
        params["lambda_e"], params["v_e"], params["delta_e"], params["e_gauge"] = optimized_params

    # 1. Extract shapes & simulation parameters
    field_shape = params["field_shape"]   # e.g. (4, 8, 8, 8)
    gauge_shape = params["gauge_shape"]     # e.g. (4, 8, 8, 8)
    grav_shape  = params["grav_shape"]      # e.g. (8, 8, 8)
    tau_end     = params["tau_end"]         # final dimensionless time
    dx          = params["dx"]              # spatial step size
    dt          = params.get("dt", 0.01)

    # Ensure PDE parameters are set (these might be overwritten by optimized params)
    params.setdefault("lambda_e", 1.0)
    params.setdefault("v_e", 1.0)
    params.setdefault("delta_e", 0.0)
    params.setdefault("e_gauge", 0.1)

    # 2. Initialize fields
    phi1_init, phi2_init = init_electron_fields(field_shape)
    A_init = init_gauge_field(gauge_shape)
    grav_init = init_gravity_field(grav_shape)

    # 3. Select PDE evolution approach
    adaptive = params.get("adaptive", False)
    if adaptive:
        dt_min        = params.get("dt_min", 1e-6)
        dt_max        = params.get("dt_max", 0.1)
        err_tolerance = params.get("err_tolerance", 1e-3)
        final_traj = evolve_fields_adaptive(
            phi1_init, phi2_init, A_init,
            tau_end, dt, dx, params,
            err_tolerance=err_tolerance,
            dt_min=dt_min,
            dt_max=dt_max
        )
    else:
        final_traj = evolve_fields(
            phi1_init, phi2_init, A_init,
            tau_end, dt, dx, params
        )

    # 4. Extract final state (time, phi1, phi2, A)
    _, phi1_fin, phi2_fin, A_fin = final_traj[-1]

    # 5. Compute final gauge field strength
    F_fin = field_strength(A_fin, dx)

    # 6. Compute diagnostics
    spin_val   = compute_spin(F_fin, dx)
    charge_val = compute_charge(F_fin, dx)
    grav_val   = compute_gravity_indentation(grav_init, dx)  # or updated via a real gravity solver

    return spin_val, charge_val, grav_val

if __name__ == "__main__":
    # Example default parameters for testing purposes.
    default_params = {
        "field_shape": (4, 16, 16, 16),
        "gauge_shape": (4, 16, 16, 16),
        "grav_shape": (16, 16, 16),
        "tau_end": 1.0,
        "dx": 0.1,
        "dt": 0.01,
        "adaptive": True,
        "dt_min": 1e-6,
        "dt_max": 0.1,
        "err_tolerance": 1e-3,
        # These will be overwritten if best_params.json exists:
        "lambda_e": 1.0,
        "v_e": 1.0,
        "delta_e": 0.0,
        "e_gauge": 0.1,
    }
    spin, charge, grav = run_dvripe_sim(default_params)
    print("Spin:", spin)
    print("Charge:", charge)
    print("Gravitational Indentation:", grav)
