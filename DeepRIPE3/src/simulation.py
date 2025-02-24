#!/usr/bin/env python3
"""
src/simulation.py

Runs the DV‑RIPE simulation using specified parameters. This module:
  1. Initializes the electron, gauge, and gravity fields.
  2. Evolves the system from τ = 0 to τ = τ_end using the implicit integrator.
  3. Computes diagnostics from the final state:
       - Effective spin,
       - Net charge, and
       - Gravitational indentation (energy proxy).

If the "verbose_integration" flag is set in the simulation parameters,
the simulation returns an integration log (a list of diagnostic snapshots at each step).
This version supports polar discretization and includes the damping parameter γ
and our full set of parameters.
"""

import os
import numpy as np
import json
import logging

# Import field initializers.
from src.fields import init_electron_fields, init_nonabelian_gauge_field, init_gravity_field
# Import the PDE solver.
from src.pde_solver import evolve_fields
# Import diagnostics.
from src.diagnostics import compute_spin, compute_charge, compute_gravity_indentation

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_dvripe_sim(params):
    """
    Main DV‑RIPE simulation function.
    
    Parameters:
      params (dict): Dictionary of simulation parameters. In polar mode, set "polar": True and supply
                     "radial_points", "angular_points", and "r_max". Other physical parameters (e.g. lambda_e,
                     v_e, delta_e, e_gauge, gamma, eta, Lambda, kappa, G_eff, mu) should also be specified.
                     If "verbose_integration" is True, the simulation returns an integration log.
                     
    Returns:
      tuple: (spin_val, charge_val, energy_proxy) if verbose_integration is False,
             or (spin_val, charge_val, energy_proxy, integration_log) if True.
    """
    # Extract grid shapes and simulation parameters.
    field_shape = params.get("field_shape", (16, 32, 64, 64))
    grav_shape  = params.get("grav_shape", (64, 64, 64))
    # Increase τ_end to allow more evolution time (adjust as needed)
    tau_end     = params.get("tau_end", 1.0)
    dx          = params.get("dx", 0.1)
    dt          = params.get("dt", 0.01)
    
    # Set default physics parameters if not provided.
    params.setdefault("lambda_e", 1.0)
    params.setdefault("v_e", 1.0)
    params.setdefault("delta_e", 0.1)
    params.setdefault("e_gauge", 0.1)
    params.setdefault("gamma", 0.0)
    params.setdefault("eta", 1.0)
    params.setdefault("Lambda", 0.5)
    params.setdefault("kappa", 1.0)
    params.setdefault("G_eff", 1.0)
    params.setdefault("mu", 0.1)
    params.setdefault("use_gpu", True)
    params.setdefault("integrator", "implicit")
    params.setdefault("mass_scale", 1.0)
    polar = params.get("polar", False)
    
    logging.info("Initializing fields:")
    logging.info("  Electron: %s  (Polar: %s)", field_shape, polar)
    logging.info("  Gravity: %s", grav_shape)
    
    # Initialize electron fields.
    if polar:
        # In polar mode, only the first two dimensions of field_shape are used for the electron grid.
        polar_shape = (field_shape[0], field_shape[1])
        phi1_init, phi2_init = init_electron_fields(polar_shape,
                                                    ambient=params.get("ambient", True),
                                                    polar=True,
                                                    phase_offset=params.get("phase_offset", 0.1),
                                                    radial_points=params.get("radial_points", 64),
                                                    angular_points=params.get("angular_points", 64),
                                                    r_max=params.get("r_max", 1.0))
    else:
        phi1_init, phi2_init = init_electron_fields(field_shape,
                                                    ambient=params.get("ambient", True),
                                                    polar=False)
    
    # Determine gauge field shape.
    if polar:
        # For polar mode, gauge_shape should be defined as (3, 4, N0, N1, radial_points, angular_points)
        N0 = field_shape[0]
        N1 = field_shape[1]
        radial_points  = params.get("radial_points", 64)
        angular_points = params.get("angular_points", 64)
        gauge_shape = (3, 4, N0, N1, radial_points, angular_points)
        logging.info("Overriding gauge_shape for polar mode: %s", gauge_shape)
    else:
        gauge_shape = params.get("gauge_shape", (3, 4, 16, 32, 64, 64))
    
    # Initialize gauge and gravity fields.
    A_init = init_nonabelian_gauge_field(gauge_shape, ambient=True)
    Phi_init = init_gravity_field(grav_shape)
    
    # In polar mode, if the gauge field has 6 dimensions, average over the angular dimension.
    if polar and A_init.ndim == 6:
        logging.info("Averaging gauge field over angular dimension for polar mode. Original shape: %s", A_init.shape)
        A_init = np.mean(A_init, axis=-1)  # New shape: (3, 4, N0, N1, radial_points)
        logging.info("New gauge field shape: %s", A_init.shape)
    
    # Evolve fields.
    final_state = evolve_fields(phi1_init, phi2_init, A_init, Phi_init, tau_end, dt, dx, params)
    
    # Check if integration_log was returned.
    if isinstance(final_state, tuple) and len(final_state) == 8:
        tau_final, phi1_fin, phi2_fin, pi1_fin, pi2_fin, A_final, Phi_final, integration_log = final_state
    else:
        tau_final, phi1_fin, phi2_fin, pi1_fin, pi2_fin, A_final, Phi_final = final_state
        integration_log = None
    
    logging.info("Simulation reached final time tau = %.4f", tau_final)
    
    # Compute diagnostics.
    spin_val = compute_spin(phi1_fin, dx)
    charge_val = compute_charge(A_final, dx)
    energy_proxy = compute_gravity_indentation(Phi_final, dx, params.get("mass_scale", 1.0))
    
    if integration_log is not None:
        return spin_val, charge_val, energy_proxy, integration_log
    else:
        return spin_val, charge_val, energy_proxy

if __name__ == "__main__":
    PARAMS_FILENAME = "electron_solution_database.json"
    if os.path.exists(PARAMS_FILENAME):
        try:
            with open(PARAMS_FILENAME, "r") as f:
                candidates = json.load(f)
                params = candidates[0]["params"]
            logging.info("Loaded parameters from %s", PARAMS_FILENAME)
        except Exception as e:
            logging.error("Error loading parameters: %s", e)
            params = {}
    else:
        params = {}
    
    # Merge with default simulation settings.
    params.setdefault("field_shape", (16, 32, 64, 64))
    params.setdefault("grav_shape", (64, 64, 64))
    params.setdefault("tau_end", 1.0)  # Increased to allow longer integration.
    params.setdefault("dx", 0.1)
    params.setdefault("dt", 0.01)
    params.setdefault("integrator", "implicit")
    params.setdefault("mass_scale", 1.0)
    params.setdefault("use_gpu", True)
    params.setdefault("polar", True)
    params.setdefault("radial_points", 64)
    params.setdefault("angular_points", 64)
    params.setdefault("r_max", 1.0)
    params.setdefault("gauge_shape", (3, 4, params["field_shape"][0],
                                      params["field_shape"][1],
                                      params.get("radial_points", 64),
                                      params.get("angular_points", 64)))
    # Enable verbose integration.
    params.setdefault("verbose_integration", True)
    
    logging.info("Using simulation parameters: %s", params)
    
    result = run_dvripe_sim(params)
    if isinstance(result, tuple) and len(result) == 4:
        spin, charge, energy, log = result
        print("Simulation complete.")
        print("Spin:", spin)
        print("Charge:", charge)
        print("Energy Proxy:", energy)
        print("Integration Log:")
        for entry in log:
            print(f"  tau={entry.get('tau', 0):.8f}, spin={entry.get('spin', 0):.4f}, charge={entry.get('charge', 0):.4f}, energy={entry.get('energy', 0):.4f}")
    else:
        spin, charge, energy = result
        print("Simulation complete.")
        print("Spin:", spin)
        print("Charge:", charge)
        print("Energy Proxy:", energy)
