#!/usr/bin/env python3
"""
simulation.py

Runs the DV-RIPE simulation using a GPU-accelerated PDE solver for the gauge field.
Scalar fields remain on the CPU while gauge field A (and its momentum E) are kept
on the GPU (if "use_gpu": True). The simulation uses ambient seeding on an enlarged domain,
and an extended simulation duration.

Diagnostics (spin, charge, gravitational indentation) are computed at the end.
"""

import numpy as np

# Field initializers
from src.fields import init_electron_fields, init_gauge_field, init_gravity_field

# PDE solvers (with GPU integration)
from src.pde_solver import evolve_fields, evolve_fields_adaptive

# Diagnostics
from src.diagnostics import compute_spin, compute_charge, compute_gravity_indentation

def run_dvripe_sim(params):
    """
    Main DV-RIPE simulation function.
    
    :param params: dict with keys:
         "field_shape", "gauge_shape", "grav_shape" (tuples)
         "tau_end", "dx" (floats)
         "dt" (float, optional; default 0.01)
         "lambda_e", "v_e", "delta_e", "e_gauge" (floats)
         "adaptive" (bool) - whether to use adaptive time stepping
         "use_gpu" (bool) - if True, gauge field evolution is performed on the GPU.
         Additional keys for adaptive stepping as needed.
    
    :return: (spin_val, charge_val, grav_val)
             Spin is computed directly from the final electron field (phi1).
    """
    
    # Extract grid shapes and parameters.
    field_shape = params["field_shape"]   # e.g. (16, 32, 64, 64)
    gauge_shape = params["gauge_shape"]     # e.g. (4, 32, 64, 64)
    grav_shape  = params["grav_shape"]      # e.g. (64, 64, 64)
    
    tau_end = params["tau_end"]
    dx = params["dx"]
    dt = params.get("dt", 0.01)
    
    # Set default PDE parameters if not provided.
    params.setdefault("lambda_e", 1.0)
    params.setdefault("v_e", 1.0)
    params.setdefault("delta_e", 0.1)
    params.setdefault("e_gauge", 0.1)
    params.setdefault("use_gpu", False)
    
    # Initialize fields.
    # Use ambient initial conditions so that the nonlinear dynamics can generate vortices.
    phi1_init, phi2_init = init_electron_fields(field_shape, ambient=True)
    A_init = init_gauge_field(gauge_shape, ambient=True)
    grav_init = init_gravity_field(grav_shape)
    
    # Choose evolution approach.
    adaptive = params.get("adaptive", False)
    if adaptive:
        dt_min = params.get("dt_min", 1e-6)
        dt_max = params.get("dt_max", 0.1)
        err_tolerance = params.get("err_tolerance", 1e-3)
        final_traj = evolve_fields_adaptive(phi1_init, phi2_init, A_init, tau_end, dt, dx, params,
                                              err_tolerance=err_tolerance, dt_min=dt_min, dt_max=dt_max)
    else:
        final_traj = evolve_fields(phi1_init, phi2_init, A_init, tau_end, dt, dx, params)
    
    # Extract final state.
    # Note: For gauge field, if GPU is used, evolve_fields() returns A as a NumPy array (via .get()).
    _, phi1_fin, phi2_fin, A_fin = final_traj[-1]
    
    # Compute diagnostics:
    # Compute spin directly from the final electron field phi1_fin.
    spin_val = compute_spin(phi1_fin, dx)
    charge_val = compute_charge(A_fin, dx)
    grav_val = compute_gravity_indentation(grav_init, dx)
    
    return spin_val, charge_val, grav_val

if __name__ == "__main__":
    # Example default parameters using the enlarged domain, extended duration, and GPU enabled.
    default_params = {
        "field_shape": (16, 32, 64, 64),    # Enlarged domain for electron fields.
        "gauge_shape": (4, 32, 64, 64),      # Enlarged gauge field.
        "grav_shape": (64, 64, 64),          # Enlarged gravity field.
        "tau_end": 5.0,                     # Extended simulation duration.
        "dx": 0.1,
        "dt": 0.01,
        "adaptive": True,
        "dt_min": 1e-6,
        "dt_max": 0.1,
        "err_tolerance": 1e-3,
        "lambda_e": 1.0,
        "v_e": 1.0,
        "delta_e": 0.1,
        "e_gauge": 0.1,
        "use_gpu": True,                    # Enable GPU acceleration.
    }
    spin, charge, grav = run_dvripe_sim(default_params)
    print("Spin:", spin)
    print("Charge:", charge)
    print("Gravitational Indentation:", grav)
