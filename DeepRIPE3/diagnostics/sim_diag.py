#!/usr/bin/env python3
"""
sim_diag.py

This diagnostic script runs the DV/RIPE simulation (using the fixed‐step solver)
with current parameters and collects detailed numerical diagnostics at each snapshot.
It writes out a text report (sim_diag_output.txt) with:
  - Time stamps.
  - Amplitude statistics (min, max, mean, std) for electron fields (phi1, phi2).
  - Phase statistics (min, max, mean) for electron fields.
  - Warnings if any NaN or Inf values are detected.
  - The computed physical diagnostics (spin, charge, gravitational indentation)
    using functions from src.diagnostics.
  
Use this output to check that nothing is going numerically awry and to understand why
the simulation might be “bottoming out.”
"""

import numpy as np
import os

# Import initializers and evolution from our project modules.
from src.fields import init_electron_fields, init_gauge_field, init_gravity_field
from src.pde_solver import evolve_fields
from src.diagnostics import compute_spin, compute_charge, compute_gravity_indentation

def run_diagnostics(params, output_filename="sim_diag_output.txt"):
    # Initialize fields
    phi1_init, phi2_init = init_electron_fields(params["field_shape"])
    A_init = init_gauge_field(params["gauge_shape"])
    grav_init = init_gravity_field(params["grav_shape"])
    
    # Run simulation evolution (fixed-step evolution for diagnostics)
    trajectory = evolve_fields(phi1_init, phi2_init, A_init, 
                               params["tau_end"], params["dt"], params["dx"], params)
    
    with open(output_filename, "w") as f:
        f.write("Simulation Diagnostics Report\n")
        f.write("=============================\n\n")
        f.write("Parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n\n")
        
        # Process each snapshot in the trajectory
        for snapshot in trajectory:
            tau, phi1, phi2, A = snapshot
            f.write(f"Time = {tau:.4f}\n")
            
            # Check for numerical issues in the fields.
            if not np.all(np.isfinite(phi1)):
                f.write("WARNING: phi1 contains NaN or Inf\n")
            if not np.all(np.isfinite(phi2)):
                f.write("WARNING: phi2 contains NaN or Inf\n")
            if not np.all(np.isfinite(A)):
                f.write("WARNING: A contains NaN or Inf\n")
            
            # Compute amplitude statistics for phi1 and phi2.
            amp1 = np.abs(phi1)
            amp2 = np.abs(phi2)
            f.write("phi1 Amplitude: min = {:.4e}, max = {:.4e}, mean = {:.4e}, std = {:.4e}\n".format(
                np.min(amp1), np.max(amp1), np.mean(amp1), np.std(amp1)))
            f.write("phi2 Amplitude: min = {:.4e}, max = {:.4e}, mean = {:.4e}, std = {:.4e}\n".format(
                np.min(amp2), np.max(amp2), np.mean(amp2), np.std(amp2)))
            
            # Compute phase statistics for phi1 and phi2.
            phase1 = np.angle(phi1)
            phase2 = np.angle(phi2)
            f.write("phi1 Phase: min = {:.4e}, max = {:.4e}, mean = {:.4e}\n".format(
                np.min(phase1), np.max(phase1), np.mean(phase1)))
            f.write("phi2 Phase: min = {:.4e}, max = {:.4e}, mean = {:.4e}\n".format(
                np.min(phase2), np.max(phase2), np.mean(phase2)))
            
            # Compute physical diagnostics (using gauge field for spin, etc.)
            spin_val = compute_spin(A, params["dx"])
            charge_val = compute_charge(A, params["dx"])
            grav_val = compute_gravity_indentation(grav_init, params["dx"])
            f.write("Diagnostics: Spin = {:.4e}, Charge = {:.4e}, Grav Indentation = {:.4e}\n".format(
                spin_val, charge_val, grav_val))
            
            f.write("-" * 50 + "\n")
    print(f"Diagnostics written to {output_filename}")

if __name__ == "__main__":
    # Default parameters for the diagnostic run. Adjust as needed.
    params = {
        "field_shape": (4, 8, 16, 16),   # (N0, N1, Ny, Nz)
        "gauge_shape": (4, 8, 16, 16),   # (4, Nx, Ny, Nz)
        "grav_shape": (16, 16, 16),       # Gravity field shape
        "tau_end": 0.5,                  # Short simulation duration for diagnostics
        "dx": 0.1,
        "dt": 0.01,
        "lambda_e": 1.0,
        "v_e": 1.0,
        "delta_e": 0.1,
        "e_gauge": 0.1,
        "adaptive": False,              # Using fixed-step evolution for diagnostics
    }
    run_diagnostics(params)
