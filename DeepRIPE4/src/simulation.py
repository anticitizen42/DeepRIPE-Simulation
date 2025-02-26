#!/usr/bin/env python3
"""
src/simulation.py
Version 2.1

Production simulation module for DV‑RIPE.
This module orchestrates a simulation run using the Radau IIA integrator.
It initializes fields using the updated fields module, allowing for perturbed initial conditions.
It computes diagnostics (effective spin, net charge, energy proxy) and logs progress.
"""

import numpy as np
import logging
from .fields import initialize_electron_field, initialize_gauge_field, initialize_gravity_field
from .pde_solver import integrate_simulation

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.propagate = False

def compute_effective_spin(electron_field):
    phase = np.angle(electron_field)
    dphase = np.diff(phase, axis=1)
    dphase = (dphase + np.pi) % (2*np.pi) - np.pi
    total_winding = np.mean(np.sum(dphase, axis=1)) / (2*np.pi)
    return abs(total_winding)

def compute_net_charge(electron_field):
    return np.sum(np.imag(electron_field)) / electron_field.size

def compute_energy_proxy(gravity_field):
    return np.mean(np.abs(gravity_field))

def run_simulation(sim_params, field_params, solver_params):
    logger.info("Starting DV‑RIPE simulation run with Radau IIA integrator.")
    
    logger.info("Initializing electron field...")
    electron_field = initialize_electron_field(
        shape=field_params.get('electron_shape', (64, 128)),
        mode=field_params.get('electron_mode', 'polar'),
        seed=field_params.get('electron_seed', 'vortex'),
        noise_level=field_params.get('noise_level', 0.01)
    )
    
    logger.info("Initializing gauge field...")
    gauge_field = initialize_gauge_field(
        grid_shape=field_params.get('gauge_grid_shape', (32, 32, 64, 128)),
        group_dim=3,
        component_dim=3,
        mode=field_params.get('gauge_mode', 'polar')
    )
    
    logger.info("Initializing gravity field...")
    gravity_field = initialize_gravity_field(
        grid_shape=field_params.get('gravity_grid_shape', (32, 32, 64, 128)),
        mode=field_params.get('gravity_mode', 'polar')
    )
    
    # Evolve the electron field; use its real part for state evolution.
    state0 = np.real(electron_field).astype(np.float32)
    
    t0 = sim_params.get('t0', 0.0)
    t_end = sim_params.get('t_end', 0.5)
    dt = sim_params.get('dt', 0.005)
    
    logger.info("Beginning integration of the PDE system...")
    times, states = integrate_simulation(state0, t0, t_end, dt, solver_params, gauge_field=gauge_field)
    
    effective_spin = compute_effective_spin(electron_field)
    net_charge = compute_net_charge(electron_field)
    energy_proxy = compute_energy_proxy(gravity_field)
    
    diagnostics = {
        'effective_spin': effective_spin,
        'net_charge': net_charge,
        'energy_proxy': energy_proxy,
        'final_time': times[-1],
        'num_steps': len(times)
    }
    
    logger.info("Simulation complete. Diagnostics:")
    logger.info(diagnostics)
    
    return diagnostics, times, states

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running test harness for DV‑RIPE simulation module (Radau IIA integrator).")
    
    simulation_parameters = {'t0': 0.0, 't_end': 0.5, 'dt': 0.005}
    field_parameters = {
        'electron_shape': (64, 128),
        'electron_mode': 'polar',
        'electron_seed': 'vortex_perturbed',  # Experiment with a perturbed vortex seed.
        'noise_level': 0.05,
        'gauge_grid_shape': (32, 32, 64, 128),
        'gauge_mode': 'polar',
        'gravity_grid_shape': (32, 32, 64, 128),
        'gravity_mode': 'polar'
    }
    solver_parameters = {
        'γ': 0.1,
        'μ': 0.05,
        'Λ': 1.0,
        'η': 0.1,
        'κ': 0.5,
        'chi': 0.5
    }
    
    diagnostics, times, states = run_simulation(simulation_parameters, field_parameters, solver_parameters)
    logger.info("Test simulation run complete.")
