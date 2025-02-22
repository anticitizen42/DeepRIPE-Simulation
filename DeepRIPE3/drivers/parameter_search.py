#!/usr/bin/env python3
"""
Adaptive Parameter Search for DV/RIPE Simulation Framework

This script uses skopt's gp_minimize to search over the parameter space,
and it automatically adjusts the search bounds if the simulation returns an error
of 1.0 (interpreted as a blow-up). The bounds are tightened around the current best
parameters to help steer the simulation into a stable region.
"""

import sys
import logging
from skopt import gp_minimize

# Import the main simulation function.
# Ensure your PYTHONPATH is set appropriately so that 'src' is in the path.
try:
    from src.simulation import run_dvripe_sim
except ImportError:
    logging.error("Could not import run_dvripe_sim from src.simulation. Check your PYTHONPATH and file structure.")
    sys.exit(1)

def params_from_list(opt_params):
    """
    Convert a list of optimized parameters into a full parameter dictionary for run_dvripe_sim.
    
    Assumes the list is in the order: [lambda, v, delta, dt_initial].
    Modify the mapping below as needed to match your simulation's expected parameters.
    """
    return {
        "field_shape": (4, 8, 8, 8),   # Fixed field shape for the simulation.
        "gauge_shape": (4, 8, 8, 8),   # Fixed gauge shape (adjust as needed).
        "grav_shape":  (8, 8, 8),       # Fixed gravity shape (adjust as needed).
        "lambda": opt_params[0],
        "v": opt_params[1],
        "delta": opt_params[2],
        "dt_initial": opt_params[3],
        "tau_end": 10.0,              # Dimensionless final time (default; adjust if needed).
        "dx": 0.1,                    # Spatial step (default; adjust as needed).
        "adaptive": True,             # Use adaptive time stepping in the simulation.
        # Add any other fixed parameters as necessary.
    }

def adaptive_objective(params, current_bounds):
    """
    Run the DV/RIPE simulation for the given parameters and adjust bounds if a blow-up occurs.
    
    Parameters:
      params (list): Current parameter values.
      current_bounds (list of tuples): The current search bounds for each parameter.
      
    Returns:
      tuple: (error, new_bounds)
             error (float): The simulation error.
             new_bounds (list of tuples): Updated bounds, tightened around `params` if error == 1.0.
    """
    # Convert list to dictionary using the helper function.
    param_dict = params_from_list(params)
    result = run_dvripe_sim(param_dict)
    # If the simulation returns a tuple, assume the first element is the error.
    if isinstance(result, tuple):
        error = result[0]
    else:
        error = result

    new_bounds = current_bounds
    if error == 1.0:
        new_bounds = []
        for i, p in enumerate(params):
            lb, ub = current_bounds[i]
            range_span = ub - lb
            # Tighten the search range around the current value by 20%
            delta_bound = 0.2 * range_span
            new_lb = max(lb, p - delta_bound)
            new_ub = min(ub, p + delta_bound)
            new_bounds.append((new_lb, new_ub))
    return error, new_bounds

def run_adaptive_parameter_search(initial_bounds, n_iterations=5, calls_per_iteration=10):
    """
    Run the parameter search with adaptive bounds.
    
    Parameters:
      initial_bounds (list of tuples): Initial bounds for each parameter.
      n_iterations (int): Maximum iterations to adjust the search space.
      calls_per_iteration (int): Function evaluations per iteration.
      
    Returns:
      tuple: (best_params, best_error, final_bounds)
    """
    bounds = initial_bounds
    best_params = None
    best_error = 1.0

    for iteration in range(n_iterations):
        logging.info(f"Iteration {iteration+1} with bounds: {bounds}")

        # Define objective for this iteration.
        def objective(params):
            err, _ = adaptive_objective(params, bounds)
            # Return a scalar value
            return float(err)

        # Run the Gaussian process minimizer with the current bounds.
        res = gp_minimize(objective, dimensions=bounds, n_calls=calls_per_iteration, random_state=iteration)
        iteration_best_error = res.fun
        iteration_best_params = res.x

        logging.info(f"Iteration {iteration+1} best params: {iteration_best_params}, error: {iteration_best_error}")

        # Update bounds based on the best parameters from this iteration.
        _, updated_bounds = adaptive_objective(iteration_best_params, bounds)
        bounds = updated_bounds

        # Break early if a stable solution is found.
        if iteration_best_error < 1.0:
            best_params = iteration_best_params
            best_error = iteration_best_error
            logging.info("Stable solution found; terminating search early.")
            break
        else:
            best_params = iteration_best_params
            best_error = iteration_best_error

    return best_params, best_error, bounds

def main():
    # Configure logging for info-level output.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Define the initial parameter search space.
    # These bounds correspond to the following parameters (in order):
    # [lambda, v, delta, dt_initial]
    initial_bounds = [
        (2.0, 6.0),   # lambda
        (0.5, 1.5),   # v
        (1.0, 5.0),   # delta
        (0.1, 0.6)    # dt_initial
    ]

    best_params, best_error, final_bounds = run_adaptive_parameter_search(initial_bounds)
    print("Final best parameters:", best_params)
    print("Final min error:", best_error)
    print("Final search bounds:", final_bounds)

if __name__ == "__main__":
    main()
