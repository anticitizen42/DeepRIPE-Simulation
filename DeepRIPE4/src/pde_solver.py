#!/usr/bin/env python3
"""
src/pde_solver.py
Version 2.3c

This module implements a fully implicit, high‑order integrator for the DV‑RIPE simulation
using the Radau IIA method with adaptive time‑stepping. This version replaces the dense
Jacobian approach with a matrix‑free Newton–Krylov solver using GMRES and employs complex‑step
differentiation for the Jacobian–vector product. The Jacobian–vector product is now wrapped
in a LinearOperator to avoid type errors and massive memory allocations.

Key features:
  - 3-stage Radau IIA method (order 5) with collocation.
  - Newton iteration with a matrix‑free, complex‑step Jacobian–vector product computed on the fly.
  - GMRES is used to solve the linear system for the Newton update, with tolerance set via 'atol'.
  - Adaptive time‑stepping via step‑doubling for error control.
  - The PDE operator includes Laplacian scaling, nonlinear potential, damping, membrane
    coupling, and inter‑field gauge coupling (scaled by cross‑talk parameter chi).

Note: For production, further enhancements (e.g., robust preconditioning, advanced Jacobian‑free methods) may be desirable.
"""

import numpy as np
import math
import logging
from scipy.sparse.linalg import gmres, LinearOperator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch_handler.setFormatter(formatter)
    logger.addHandler(ch_handler)

# Basic PDE operator functions.
def compute_laplacian(field):
    return (np.roll(field, 1, axis=-2) +
            np.roll(field, -1, axis=-2) +
            np.roll(field, 1, axis=-1) +
            np.roll(field, -1, axis=-1) -
            4.0 * field)

def compute_potential_derivative(field, parameters):
    Lambda = parameters.get('Λ', 1.0)
    eta = parameters.get('η', 1.0)
    potential_scale = (1.0 + Lambda) * eta
    return potential_scale * (field ** 3)

def gauge_covariant_derivative(field, gauge_field, parameters):
    dfdx = (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1)) / 2.0
    dfdy = (np.roll(field, -1, axis=-2) - np.roll(field, 1, axis=-2)) / 2.0
    simple_deriv = dfdx + dfdy
    chi = parameters.get('chi', 0.0)
    return chi * simple_deriv

def pde_operator(state, parameters, gauge_field=None):
    gamma = parameters.get('γ', 0.1)
    mu = parameters.get('μ', 0.0)
    kappa = parameters.get('κ', 1.0)
    laplacian_term = (1.0 + mu) * compute_laplacian(state)
    potential_term = compute_potential_derivative(state, parameters)
    damping_term = -gamma * state + kappa * state
    if gauge_field is not None:
        gauge_term = gauge_covariant_derivative(state, gauge_field, parameters)
    else:
        gauge_term = 0.0
    return damping_term + laplacian_term + potential_term + gauge_term

def radau_iia_step(state, dt, parameters, gauge_field=None, newton_tol=1e-6, newton_max_iter=50):
    """
    Perform one step of the 3-stage Radau IIA method using a matrix-free Newton–Krylov method.
    
    Solves for stage values Y_i (i=1,...,3) satisfying:
      Y_i = state + dt * sum_{j=1}^{3} A_{ij} f(Y_j)
    where f(Y) = pde_operator(Y, parameters, gauge_field).
    
    Then the new state is computed as:
      new_state = state + dt * sum_{j=1}^{3} b_j f(Y_j)
    
    Uses GMRES to solve the linear system for the Newton update with a complex‑step
    Jacobian–vector product wrapped in a LinearOperator.
    
    Returns:
      new_state (np.ndarray): The updated state.
    """
    # 3-stage Radau IIA coefficients.
    sqrt6 = math.sqrt(6)
    c = np.array([(4 - sqrt6) / 10, (4 + sqrt6) / 10, 1.0])
    A = np.array([
        [(88 - 7 * sqrt6) / 360, (296 - 169 * sqrt6) / 1800, (-2 + 3 * sqrt6) / 225],
        [(296 + 169 * sqrt6) / 1800, (88 + 7 * sqrt6) / 360, (-2 - 3 * sqrt6) / 225],
        [(16 - sqrt6) / 36, (16 + sqrt6) / 36, 1.0/9.0]
    ])
    b = np.array([(16 - sqrt6) / 36, (16 + sqrt6) / 36, 1.0/9.0])
    
    s = 3  # number of stages.
    # Initialize stage values as copies of state, cast to complex.
    Y = np.array([state.copy() for _ in range(s)], dtype=np.float64).astype(np.complex128)
    state_complex = state.astype(np.complex128)
    N = state.size
    
    # Define the residual function F for Newton iteration in complex arithmetic.
    def F(Y_flat):
        Y_stages = Y_flat.reshape((s,) + state.shape).astype(np.complex128)
        F_val = np.empty_like(Y_stages, dtype=np.complex128)
        for i in range(s):
            sum_term = np.zeros_like(state_complex, dtype=np.complex128)
            for j in range(s):
                sum_term += A[i, j] * pde_operator(Y_stages[j], parameters, gauge_field).astype(np.complex128)
            F_val[i] = Y_stages[i] - state_complex - dt * sum_term
        return F_val.flatten()
    
    # Flatten initial guess.
    Y_flat = Y.flatten()
    n = Y_flat.size
    
    # Define the Jacobian-vector product using complex‑step differentiation.
    def Jv(v):
        epsilon = 1e-20
        return (F(Y_flat + 1j * epsilon * v) - F(Y_flat)) / (1j * epsilon)
    
    # Wrap Jv in a LinearOperator.
    A_operator = LinearOperator((n, n), matvec=Jv, dtype=np.complex128)
    
    # Newton iteration using GMRES to solve for delta.
    for iter in range(newton_max_iter):
        F_val = F(Y_flat)
        normF = np.linalg.norm(F_val)
        if normF < newton_tol:
            logger.debug(f"Newton iteration converged in {iter+1} iterations (||F||={normF:.2e}).")
            break
        delta, exitCode = gmres(A_operator, -F_val, atol=newton_tol)
        if exitCode != 0:
            logger.error(f"GMRES did not converge at Newton iteration {iter+1} (exit code {exitCode}).")
            break
        Y_flat = Y_flat + delta
    else:
        logger.warning("Newton iteration did not converge within the maximum number of iterations.")
    
    Y = Y_flat.reshape((s,) + state.shape)
    new_state = state_complex.copy()
    for j in range(s):
        new_state += dt * b[j] * pde_operator(Y[j], parameters, gauge_field).astype(np.complex128)
    return np.real(new_state).astype(state.dtype)

def radau_iia_step_with_error(state, dt, parameters, gauge_field=None, newton_tol=1e-6, newton_max_iter=50):
    """
    Compute one full step using Radau IIA and estimate the error using step doubling.
    
    Returns:
      new_state: solution after time dt
      error_est: estimated error norm between one full step and two half-steps.
    """
    full_step = radau_iia_step(state, dt, parameters, gauge_field, newton_tol, newton_max_iter)
    half_step = radau_iia_step(state, dt/2, parameters, gauge_field, newton_tol, newton_max_iter)
    two_half_steps = radau_iia_step(half_step, dt/2, parameters, gauge_field, newton_tol, newton_max_iter)
    error_est = np.linalg.norm(full_step - two_half_steps)
    return two_half_steps, error_est

def integrate_simulation(state0, t0, t_end, dt, parameters, gauge_field=None,
                         dt_min=1e-8, max_steps=10000, tol=1e-4):
    """
    Integrate the system using the Radau IIA method with adaptive time-stepping.
    
    The step-doubling error estimate is used to adapt dt.
    
    Returns:
      times: list of time stamps.
      states: list of state snapshots.
    """
    times = [t0]
    states = [state0.copy()]
    t = t0
    state = state0.copy()
    step = 0
    while t < t_end and step < max_steps:
        new_state, err = radau_iia_step_with_error(state, dt, parameters, gauge_field)
        if err > tol:
            logger.warning(f"High error {err:.2e} at t={t:.4e}; reducing dt.")
            dt *= 0.5
            if dt < dt_min:
                logger.error("dt below dt_min. Aborting integration.")
                break
            continue
        t += dt
        state = new_state.copy()
        times.append(t)
        states.append(state.copy())
        step += 1
        logger.info(f"t = {t:.4e}, dt = {dt:.2e}, error_est = {err:.2e}, state norm = {np.linalg.norm(state):.4e}")
        if err < tol/10:
            dt = min(dt * 1.5, (t_end - t))
    if step >= max_steps:
        logger.error("Maximum number of steps reached before t_end.")
    else:
        logger.info("Integration complete.")
    return times, states

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running test harness for Radau IIA integrator with complex‑step Jacobian (matrix-free Newton–Krylov).")
    
    # Create a test 2D field representing a vortex in polar coordinates.
    test_shape = (128, 256)
    r = np.linspace(0, 1, test_shape[0])
    theta = np.linspace(0, 2 * np.pi, test_shape[1], endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    amplitude = np.exp(-R)
    phase = Theta
    initial_state = amplitude * np.exp(1j * phase)
    state0 = np.real(initial_state).astype(np.float32)
    
    parameters = {
        'γ': 0.1,
        'μ': 0.05,
        'Λ': 1.0,
        'η': 0.1,
        'κ': 0.5,
        'chi': 0.5
    }
    
    t0 = 0.0
    t_end = 0.1
    dt = 0.005
    times, states = integrate_simulation(state0, t0, t_end, dt, parameters, dt_min=1e-8, tol=1e-4)
    logger.info(f"Test integration produced {len(times)} time steps.")
