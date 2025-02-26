#!/usr/bin/env python3
"""
drivers/parameter_scanner.py
Version 2.5

Automatic Parameter Scanner for DV‑RIPE Simulation with Enhanced Diagnostics.
This script varies key solver parameters by ±30% from a base candidate (using factors [0.7, 1.0, 1.3]),
runs the simulation for each candidate using the Radau IIA integrator,
and evaluates emergent observables against target electron properties:
  - Effective spin ≈ 0.5
  - Energy proxy ≈ 1.0
  - Net charge ≈ -1.0

Additionally, it computes upgraded diagnostics including:
  - The peak spectral energy from a 2D Fourier transform of the final state.
  - The spatial standard deviation of the final state.

The integration time is extended (t_end = 1.0) and the initial electron seed is set to "vortex_perturbed" with a higher noise level.
Final diagnostics are saved to disk.
"""

import numpy as np
import itertools
import logging
import gc
import os
from src.simulation import run_simulation

# Clear any existing handlers.
logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Target diagnostics.
TARGET_SPIN = 0.5
TARGET_ENERGY = 1.0
TARGET_CHARGE = -1.0

SPIN_TOL = 0.05
ENERGY_TOL = 0.1
CHARGE_TOL = 0.1

# Base solver parameters, including cross‑talk.
base_solver_params = {
    'γ': 0.1,
    'μ': 0.05,
    'Λ': 1.0,
    'η': 0.1,
    'κ': 0.5,
    'chi': 0.5
}

# Broaden variation factors: ±30%
variation_factors = [0.7, 1.0, 1.3]
scan_keys = list(base_solver_params.keys())

def generate_parameter_combinations(base_params, keys, factors):
    varied_values = { key: [base_params[key] * f for f in factors] for key in keys }
    all_combinations = list(itertools.product(*(varied_values[key] for key in keys)))
    param_dicts = []
    for combo in all_combinations:
        params = base_params.copy()
        for i, key in enumerate(keys):
            params[key] = combo[i]
        param_dicts.append(params)
    return param_dicts

def compute_additional_diagnostics(final_state):
    """
    Compute extra diagnostics from the final state:
      - peak_spectral_energy: the maximum squared magnitude from a 2D FFT.
      - state_std: standard deviation of the field.
    """
    fft_field = np.fft.fftshift(np.fft.fft2(final_state))
    peak_spectral_energy = np.max(np.abs(fft_field)**2) / final_state.size
    state_std = np.std(final_state)
    return {"peak_spectral_energy": peak_spectral_energy, "state_std": state_std}

def self_diagnostics(diagnostics, times, states):
    passed = True
    for key in ['effective_spin', 'net_charge', 'energy_proxy', 'final_time', 'num_steps']:
        value = diagnostics.get(key, None)
        if value is None:
            logger.error(f"Self-diagnostics failed: Missing key '{key}'.")
            passed = False
        elif not np.isfinite(value):
            logger.error(f"Self-diagnostics failed: '{key}' is not finite (value: {value}).")
            passed = False
    if len(times) < 2 or len(states) < 2:
        logger.error("Self-diagnostics failed: Insufficient integration steps.")
        passed = False
    expected_t_end = diagnostics.get("final_time", None)
    if expected_t_end is None or abs(expected_t_end - times[-1]) > 1e-6:
        logger.error(f"Self-diagnostics failed: Final time mismatch (times[-1]={times[-1]}, expected {expected_t_end}).")
        passed = False
    return passed

def run_scanner():
    logger.info("Starting parameter scan with enhanced diagnostics (Radau IIA integrator)...")
    
    candidate_params = generate_parameter_combinations(base_solver_params, scan_keys, variation_factors)
    logger.info(f"Generated {len(candidate_params)} candidate parameter sets.")
    
    # Extended integration time for better long-term dynamics.
    sim_params = {'t0': 0.0, 't_end': 1.0, 'dt': 0.005}
    field_params = {
        'electron_shape': (64, 128),
        'electron_mode': 'polar',
        'electron_seed': 'vortex_perturbed',  # Use perturbed initial condition.
        'noise_level': 0.1,
        'gauge_grid_shape': (32, 32, 64, 128),
        'gauge_mode': 'polar',
        'gravity_grid_shape': (32, 32, 64, 128),
        'gravity_mode': 'polar'
    }
    successful_candidates = []
    diagnostics_list = []
    
    for i, params in enumerate(candidate_params):
        logger.info(f"Running candidate {i+1}/{len(candidate_params)} with parameters: {params}")
        try:
            diagnostics, times, states = run_simulation(sim_params, field_params, params)
        except Exception as e:
            logger.error(f"Simulation failed for candidate {i+1}: {e}")
            continue
        if not self_diagnostics(diagnostics, times, states):
            logger.error(f"Self-diagnostics failed for candidate {i+1}. Skipping.")
            del times, states
            gc.collect()
            continue
        
        # Upgrade diagnostics: Compute additional measures from final state.
        final_state = states[-1]
        extra_diag = compute_additional_diagnostics(final_state)
        diagnostics.update(extra_diag)
        
        spin = diagnostics.get('effective_spin', None)
        energy = diagnostics.get('energy_proxy', None)
        charge = diagnostics.get('net_charge', None)
        logger.info(f"Candidate {i+1} diagnostics: Spin = {spin:.4f}, Energy = {energy:.4f}, Charge = {charge:.4f}, " +
                    f"Peak Spectrum = {diagnostics.get('peak_spectral_energy', np.nan):.4e}, State STD = {diagnostics.get('state_std', np.nan):.4e}")
        
        # Decide if candidate meets target diagnostics (only global ones used for now).
        if (spin is not None and abs(spin - TARGET_SPIN) <= SPIN_TOL and
            energy is not None and abs(energy - TARGET_ENERGY) <= ENERGY_TOL and
            charge is not None and abs(charge - TARGET_CHARGE) <= CHARGE_TOL):
            logger.info(f"Candidate {i+1} meets target diagnostics!")
            successful_candidates.append((params, diagnostics))
        else:
            logger.info(f"Candidate {i+1} does not meet target diagnostics.")
        diagnostics_list.append({'candidate': i+1, 'params': params, 'diagnostics': diagnostics})
        del times, states
        gc.collect()
    
    results_file = "enhanced_diagnostics_results.npy"
    np.save(results_file, diagnostics_list)
    logger.info(f"Saved final diagnostics of all candidates to {results_file}")
    logger.info(f"Parameter scan complete. {len(successful_candidates)} candidates met target diagnostics.")
    for idx, (params, diag) in enumerate(successful_candidates):
        logger.info(f"Success #{idx+1}: Parameters: {params}, Diagnostics: {diag}")
    
    return successful_candidates

if __name__ == "__main__":
    import gc
    run_scanner()
