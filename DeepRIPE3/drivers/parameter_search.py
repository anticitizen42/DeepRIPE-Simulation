# src/drivers/parameter_search.py

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from src.simulation import run_dvripe_sim

search_space = [
    Real(0.0, 5.0, name="lambda_e"),
    Real(0.0, 5.0, name="v_e"),
    Real(0.0, 5.0, name="delta_e"),
    Real(0.0, 1.0, name="e_gauge")
]

TARGET_SPIN   = 0.5
TARGET_CHARGE = -1.0
TARGET_GRAV   = 1.0

@use_named_args(search_space)
def objective(lambda_e, v_e, delta_e, e_gauge):
    params = {
        "field_shape": (4,8,8,8),
        "gauge_shape": (4,8,8,8),
        "grav_shape": (8,8,8),
        "tau_end": 1.0,
        "dtau": 0.01,
        "dx": 1.0,
        "adaptive": True,
        "lambda_e": lambda_e,
        "v_e": v_e,
        "delta_e": delta_e,
        "e_gauge": e_gauge
    }
    try:
        spin_val, charge_val, grav_val = run_dvripe_sim(params)
        e1 = abs(spin_val - TARGET_SPIN)/abs(TARGET_SPIN)
        e2 = abs(charge_val - TARGET_CHARGE)/abs(TARGET_CHARGE)
        e3 = abs(grav_val - TARGET_GRAV)
        return e1 + e2 + e3
    except:
        return 1e6

def main():
    result = gp_minimize(func=objective, dimensions=search_space, n_calls=20, n_random_starts=5)
    print("Best parameters:", result.x)
    print("Min error:", result.fun)

if __name__=="__main__":
    main()
