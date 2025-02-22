#!/usr/bin/env python3
"""
simulation.py

Runs the DV/RIPE simulation, calling either the fixed-step or adaptive-step PDE solver
in pde_solver.py. Also measures spin, charge, and gravitational indentation at the end.
"""

import numpy as np

# Field initializers
from src.fields import init_electron_fields, init_gauge_field, init_gravity_field

# PDE solvers (fixed + adaptive) and gauge utilities
from src.pde_solver import (
    evolve_fields,
    evolve_fields_adaptive,
    field_strength  # no longer used for spin computation
)

# Diagnostics
from src.diagnostics import (
    compute_spin,            # now compute spin directly from phi1
    compute_charge,
    compute_gravity_indentation
)


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

    # 1. Extract shapes & simulation parameters
    field_shape = params["field_shape"]   # e.g. (4, 8, 8, 8) or (4, 8, 16, 16)
    gauge_shape = params["gauge_shape"]     # e.g. (4, 8, 16, 16)
    grav_shape  = params["grav_shape"]      # e.g. (16, 16, 16)

    tau_end = params["tau_end"]             # dimensionless final time
    dx      = params["dx"]                  # spatial step

    # Provide a default dt if not specified
    dt = params.get("dt", 0.01)

    # PDE parameters for electron fields (adapt for protons if needed)
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

    # 4. Extract final state
    _, phi1_fin, phi2_fin, A_fin = final_traj[-1]

    # 5. Compute diagnostics:
    # Now compute spin directly from the final electron field (phi1_fin).
    spin_val   = compute_spin(phi1_fin, dx)
    charge_val = compute_charge(A_fin, dx)
    # For gravity, we use the initial gravity field (could be updated with a proper solver)
    grav_val   = compute_gravity_indentation(grav_init, dx)

    return spin_val, charge_val, grav_val


if __name__ == "__main__":
    # Example default parameters for testing purposes.
    default_params = {
        "field_shape": (4, 8, 16, 16),
        "gauge_shape": (4, 8, 16, 16),
        "grav_shape": (16, 16, 16),
        "tau_end": 1.0,
        "dx": 0.1,
        "dt": 0.01,
        "adaptive": True,
        "dt_min": 1e-6,
        "dt_max": 0.1,
        "err_tolerance": 1e-3,
        # These will be used (or overwritten) by parameter search if needed:
        "lambda_e": 1.0,
        "v_e": 1.0,
        "delta_e": 0.0,
        "e_gauge": 0.1,
    }
    spin, charge, grav = run_dvripe_sim(default_params)
    print("Spin:", spin)
    print("Charge:", charge)
    print("Gravitational Indentation:", grav)
