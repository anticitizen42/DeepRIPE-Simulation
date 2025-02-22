#!/usr/bin/env python3
"""
pde_solver.py

Implements a second-order leapfrog PDE solver for two scalar fields (phi1, phi2)
coupled to a gauge field A that now evolves dynamically with its conjugate momentum E.
This version includes:
  - Finite-difference operators for 4D and 3D fields.
  - A naive covariant derivative for the scalar fields.
  - An updated potential derivative (Mexican-hat/Abelian Higgs) that supports vortex formation.
  - Gauge dynamics: evolution of A and its momentum E using a simple wave-like operator.
  - Fixed-step and adaptive-step evolution routines.
"""

import numpy as np

###############################################################################
# 1. Finite-Difference & Gauge Utilities
###############################################################################

def finite_difference_laplacian(field, dx):
    """
    Compute the Laplacian using a central finite-difference scheme.
    Assumes the field is a 4D array, e.g. with shape (Nx, Ny, Nz, Nu).
    """
    lap = np.zeros_like(field, dtype=field.dtype)
    if all(s >= 3 for s in field.shape):
        lap[1:-1, 1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1, 1:-1] +
            field[1:-1, 2:, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1, 1:-1] +
            field[1:-1, 1:-1, 2:, 1:-1] + field[1:-1, 1:-1, :-2, 1:-1] +
            field[1:-1, 1:-1, 1:-1, 2:] + field[1:-1, 1:-1, 1:-1, :-2] -
            8.0 * field[1:-1, 1:-1, 1:-1, 1:-1]
        ) / (dx**2)
    return lap

def finite_difference_laplacian_3d(field, dx):
    """
    Compute the Laplacian on a 3D field using a central finite-difference scheme.
    Assumes field has shape (Nx, Ny, Nz).
    """
    lap = np.zeros_like(field, dtype=field.dtype)
    if all(s >= 3 for s in field.shape):
        lap[1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
            field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
            field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
            6.0 * field[1:-1, 1:-1, 1:-1]
        ) / (dx**2)
    return lap

def gauge_derivative(phi, A, e_charge, dx):
    """
    Compute a naive covariant derivative for phi:
       D(phi) = ∂(phi) - i * e_charge * A * phi.
    Uses np.gradient along axis=0 as a placeholder.
    """
    dphi = np.gradient(phi, dx, axis=0)
    gauge_term = -1j * e_charge * A[0] * phi  # using the first component of A as a placeholder
    return dphi + gauge_term

def gauge_potential_sq(A):
    """
    Compute the squared magnitude of the gauge field A.
    A is expected to have shape (4, Nx, Ny, Nz).
    """
    return np.sum(A**2, axis=0)

def field_strength(A, dx):
    """
    Compute the field strength tensor F[mu,nu] = ∂_mu A_nu - ∂_nu A_mu.
    A is expected to have shape (4, Nx, Ny, Nz), with index 0 as time and indices 1,2,3 as spatial.
    For time derivatives we assume a static gauge field (i.e. ∂_t A = 0).
    Returns an array F of shape (4, 4, Nx, Ny, Nz).
    """
    shape = A.shape  # (4, Nx, Ny, Nz)
    F = np.zeros((4, 4) + shape[1:], dtype=A.dtype)

    def spatial_grad(comp, spatial_axis):
        return np.gradient(comp, dx, axis=spatial_axis)

    for mu in range(4):
        for nu in range(mu+1, 4):
            if mu == 0 or nu == 0:
                grad_A_nu = np.zeros_like(A[nu])
                grad_A_mu = np.zeros_like(A[mu])
            else:
                grad_A_nu = spatial_grad(A[nu], mu - 1)
                grad_A_mu = spatial_grad(A[mu], nu - 1)
            F[mu, nu] = grad_A_nu - grad_A_mu
            F[nu, mu] = -F[mu, nu]
    return F

def gauge_operator(A, dx):
    """
    A simple operator for gauge dynamics.
    In vacuum, Maxwell's equations (in Lorenz gauge) imply that A satisfies a wave equation:
       ∂_t^2 A = ∇^2 A  (ignoring current).
    Here we compute the Laplacian of each spatial component of A.
    The time component (index 0) is set to zero (assuming temporal gauge).
    """
    op = np.zeros_like(A, dtype=A.dtype)
    op[0] = 0  # time component remains unchanged
    for i in range(1, 4):
        op[i] = finite_difference_laplacian_3d(A[i], dx)
    return op

###############################################################################
# 2. Potential & PDE Operators (Updated to Support Vortex Formation)
###############################################################################

def potential_derivative(phi, phi_other, lambda_, v, delta):
    """
    Compute dV/dphi for a potential that supports vortex formation.
    
    We use a Mexican-hat (Abelian Higgs) potential combined with an inter-field coupling:
    
       V(phi, phi_other) = (lambda/4) * (|phi|^2 - v^2)^2 + delta * Re(phi * phi_other^*)
    
    Its derivative with respect to phi is:
    
       dV/dphi = (lambda/2) * (|phi|^2 - v^2) * phi + delta * (phi - phi_other)
       
    """
    return 0.5 * lambda_ * (np.abs(phi)**2 - v**2) * phi + delta * (phi - phi_other)

def covariant_laplacian(phi, A, e_charge, dx):
    """
    Compute the covariant Laplacian on phi:
       D^mu D_mu phi = finite_difference_laplacian(phi) + gauge_derivative(phi)
    """
    lap = finite_difference_laplacian(phi, dx)
    gauge_term = gauge_derivative(phi, A, e_charge, dx)
    return lap + gauge_term

def PDE_operator(phi, phi_other, A, e_charge, dx, lambda_, v, delta):
    """
    Full PDE operator:
       D^mu D_mu phi - dV/dphi,
    where dV/dphi is computed using the updated Mexican-hat potential.
    """
    lap_term = covariant_laplacian(phi, A, e_charge, dx)
    pot_term = potential_derivative(phi, phi_other, lambda_, v, delta)
    return lap_term - pot_term

###############################################################################
# 3. Leapfrog Step & Fixed-Step Evolution (Including Gauge Dynamics)
###############################################################################

def leapfrog_step(phi1, pi1, phi2, pi2, A, E, dt, dx, params):
    """
    Perform one leapfrog time step for:
      - Scalar fields phi1, phi2 with conjugate momenta pi1, pi2.
      - Gauge field A with conjugate momentum E.
      
    Scalar fields are updated using the PDE operator.
    The gauge field is updated using a simple wave equation operator.
    """
    e_charge = params["e_gauge"]
    lambda_e = params["lambda_e"]
    v_e = params["v_e"]
    delta_e = params["delta_e"]

    # --- Scalar Field Update ---
    # First half-step for scalar momenta:
    pi1_half = pi1 + 0.5 * dt * PDE_operator(phi1, phi2, A, e_charge, dx, lambda_e, v_e, delta_e)
    pi2_half = pi2 + 0.5 * dt * PDE_operator(phi2, phi1, A, e_charge, dx, lambda_e, v_e, delta_e)
    # Full-step update for scalar fields:
    phi1_new = phi1 + dt * pi1_half
    phi2_new = phi2 + dt * pi2_half
    # Second half-step for scalar momenta:
    pi1_new = pi1_half + 0.5 * dt * PDE_operator(phi1_new, phi2_new, A, e_charge, dx, lambda_e, v_e, delta_e)
    pi2_new = pi2_half + 0.5 * dt * PDE_operator(phi2_new, phi1_new, A, e_charge, dx, lambda_e, v_e, delta_e)

    # --- Gauge Field Update ---
    # Update gauge momentum E using a simple operator (vacuum wave equation):
    E_half = E + 0.5 * dt * gauge_operator(A, dx)
    A_new = A + dt * E_half
    E_new = E_half + 0.5 * dt * gauge_operator(A_new, dx)

    return phi1_new, pi1_new, phi2_new, pi2_new, A_new, E_new

def evolve_fields(phi1_init, phi2_init, A_init, tau_end, dt, dx, params):
    """
    Fixed-step evolution using the leapfrog scheme from t=0 to tau_end,
    including scalar and gauge field dynamics.
    Returns a list of snapshots (tau, phi1, phi2, A).
    """
    shape = phi1_init.shape
    # Initialize scalar momenta
    pi1 = np.zeros(shape, dtype=phi1_init.dtype)
    pi2 = np.zeros(shape, dtype=phi1_init.dtype)
    # Initialize gauge momentum E (same shape as A)
    E = np.zeros_like(A_init, dtype=A_init.dtype)

    phi1 = phi1_init.copy()
    phi2 = phi2_init.copy()
    A = A_init.copy()

    tau = 0.0
    snapshots = []
    while tau < tau_end:
        snapshots.append((tau, phi1.copy(), phi2.copy(), A.copy()))
        if tau + dt > tau_end:
            dt = tau_end - tau
        phi1, pi1, phi2, pi2, A, E = leapfrog_step(phi1, pi1, phi2, pi2, A, E, dt, dx, params)
        tau += dt
    snapshots.append((tau, phi1.copy(), phi2.copy(), A.copy()))
    return snapshots

###############################################################################
# 4. Adaptive-Step Evolution (Including Gauge Dynamics)
###############################################################################

def local_error_norm(phi1_big, pi1_big, phi2_big, pi2_big, A_big, E_big,
                     phi1_small, pi1_small, phi2_small, pi2_small, A_small, E_small):
    """
    Compute a norm representing the difference between one full step and two half-steps.
    Includes contributions from scalar fields and the gauge field (A) and its momentum (E).
    """
    err_phi1 = np.sqrt(np.sum(np.abs(phi1_big - phi1_small)**2))
    err_phi2 = np.sqrt(np.sum(np.abs(phi2_big - phi2_small)**2))
    err_A = np.sqrt(np.sum(np.abs(A_big - A_small)**2))
    err_E = np.sqrt(np.sum(np.abs(E_big - E_small)**2))
    return err_phi1 + err_phi2 + err_A + err_E

def evolve_fields_adaptive(phi1_init, phi2_init, A_init, tau_end, dt_initial, dx, params,
                           err_tolerance=1e-3, dt_min=1e-12, dt_max=0.1):
    """
    Adaptive evolution using the leapfrog scheme.
    Compares one full step (dt) against two half-steps (dt/2 each) to estimate local error.
    Adjusts dt dynamically to keep the local error below err_tolerance.
    Returns a list of snapshots (tau, phi1, phi2, A).
    """
    shape = phi1_init.shape
    pi1 = np.zeros(shape, dtype=phi1_init.dtype)
    pi2 = np.zeros(shape, dtype=phi1_init.dtype)
    E = np.zeros_like(A_init, dtype=A_init.dtype)

    phi1 = phi1_init.copy()
    phi2 = phi2_init.copy()
    A = A_init.copy()

    tau = 0.0
    dt = dt_initial
    snapshots = []
    
    while tau < tau_end:
        snapshots.append((tau, phi1.copy(), phi2.copy(), A.copy()))
        if tau + dt > tau_end:
            dt = tau_end - tau
            if dt < dt_min:
                print("Adaptive solver: dt_min reached near end. Stopping.")
                break

        # One full step of size dt
        phi1_big, pi1_big, phi2_big, pi2_big, A_big, E_big = leapfrog_step(phi1, pi1, phi2, pi2, A, E, dt, dx, params)
        # Two half-steps of size dt/2
        half_dt = 0.5 * dt
        phi1_half, pi1_half, phi2_half, pi2_half, A_half, E_half = leapfrog_step(phi1, pi1, phi2, pi2, A, E, half_dt, dx, params)
        phi1_small, pi1_small, phi2_small, pi2_small, A_small, E_small = leapfrog_step(phi1_half, pi1_half, phi2_half, pi2_half, A_half, E_half, half_dt, dx, params)
        
        err = local_error_norm(phi1_big, pi1_big, phi2_big, pi2_big, A_big, E_big,
                               phi1_small, pi1_small, phi2_small, pi2_small, A_small, E_small)
        
        if not np.isfinite(err) or err > err_tolerance:
            dt *= 0.5
            if dt < dt_min:
                print("Adaptive solver: dt_min exceeded due to error. Stopping.")
                break
            continue  # Retry with a smaller dt
        else:
            phi1, pi1, phi2, pi2, A, E = phi1_small, pi1_small, phi2_small, pi2_small, A_small, E_small
            tau += dt
            if err < 0.1 * err_tolerance and dt < dt_max:
                dt *= 1.2
    snapshots.append((tau, phi1.copy(), phi2.copy(), A.copy()))
    return snapshots
