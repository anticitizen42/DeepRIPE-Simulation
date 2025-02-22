#!/usr/bin/env python3

"""
test_dvripe_simulation.py

A minimal test that runs the DV/RIPE simulation for a few steps,
then prints the resulting spin, charge, and gravitational indentation.
"""

from src.simulation import run_dvripe_sim

def main():
    params = {
        "field_shape": (4, 8, 8, 8),
        "gauge_shape": (4, 8, 8, 8),
        "grav_shape":  (8, 8, 8),
        "tau_end": 2.0,
        "dtau": 0.01,
        "dx": 1.0,
        "lambda_e": 1.0,
        "v_e": 1.0,
        "delta_e": 1.0,
        "e_gauge": 0.1
    }
    spin_val, charge_val, grav_val = run_dvripe_sim(params)
    print("=== Minimal DV/RIPE Simulation Test ===")
    print(f"Spin:  {spin_val}")
    print(f"Charge:{charge_val}")
    print(f"Grav:  {grav_val}")

if __name__ == "__main__":
    main()
