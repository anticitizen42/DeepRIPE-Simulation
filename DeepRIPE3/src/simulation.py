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
    field_strength
)

# Diagnostics
from src.diagnostics import (
    compute_spin,
    compute_charge,
    compute_gravity_indentation
)


def run_dvripe_sim(params):
    """
    Main DV/RIPE simulation function.

    :param params: dict with keys:
       "field_shape", "gauge_shape", "grav_shape" (tuples)
       "tau_end", "dx" (floats)
       "dt" (float, optional) - if missing, we default to 0.01
       "lambda_e", "v_e", "delta_e", "e_gauge" (floats)
       "adaptive" (bool) - whether to use adaptive time stepping
       "dt_min", "dt_max", "err_tolerance" (for adaptive stepping)
    :return: (spin_val, charge_val, grav_val)
    """

    # 1. Extract shapes & simulation parameters
    field_shape = params["field_shape"]   # e.g. (4, 8, 8, 8)
    gauge_shape = params["gauge_shape"]   # e.g. (4, 8, 8, 8)
    grav_shape  = params["grav_shape"]    # e.g. (8, 8, 8)

    tau_end = params["tau_end"]           # dimensionless final time
    dx      = params["dx"]                # spatial step

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
        # Use fixed-step approach
        final_traj = evolve_fields(
            phi1_init, phi2_init, A_init,
            tau_end, dt, dx, params
        )

    # 4. Extract final state
    _, phi1_fin, phi2_fin, A_fin = final_traj[-1]

    # 5. Compute final gauge field strength
    F_fin = field_strength(A_fin, dx)

    # 6. Measure spin, charge, gravitational indentation
    spin_val  = compute_spin(F_fin, dx)
    charge_val= compute_charge(F_fin, dx)
    grav_val  = compute_gravity_indentation(grav_init, dx)  # or update gravity if you do a real solver

    return spin_val, charge_val, grav_val
