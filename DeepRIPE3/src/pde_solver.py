#!/usr/bin/env python3
"""
pde_solver.py

Implements a second-order leapfrog PDE solver for two scalar fields (phi1, phi2)
coupled to a gauge field A. Includes both a fixed-step solver (evolve_fields)
and an adaptive-step solver (evolve_fields_adaptive).

You should adapt/replace the placeholders (potential_derivative, gauge_derivative,
etc.) to match your final DV/RIPE physics.
"""

import numpy as np

###############################################################################
# 1. Finite-Difference & Gauge Utilities
###############################################################################

def finite_difference_laplacian(field, dx):
    """
    4D array: field[t,x,y,z], or if your shape is (Nx,Ny,Nz) that's also fine.
    We'll do a naive approach for a 4D shape: field.shape=(Nx,Ny,Nz,Nu).
    If you have only 3D, adapt the slicing accordingly.

    This version checks that shape dims >= 2, then does a 4D lap in x,y,z,u.
    You can adapt for a 3D domain if needed.
    """
    lap = np.zeros_like(field, dtype=field.dtype)
    if all(s>=3 for s in field.shape):
        lap[1:-1,1:-1,1:-1,1:-1] = (
            field[2:,1:-1,1:-1,1:-1] + field[:-2,1:-1,1:-1,1:-1] +
            field[1:-1,2:,1:-1,1:-1] + field[1:-1,:-2,1:-1,1:-1] +
            field[1:-1,1:-1,2:,1:-1] + field[1:-1,1:-1,:-2,1:-1] +
            field[1:-1,1:-1,1:-1,2:] + field[1:-1,1:-1,1:-1,:-2] -
            8.0*field[1:-1,1:-1,1:-1,1:-1]
        ) / (dx**2)
    return lap

def gauge_derivative(phi, A, e_charge, dx):
    """
    Covariant derivative placeholder:
    D^mu phi ~ partial^mu phi - i e_charge A^mu phi
    We'll do a naive partial derivative along axis=0 minus i e A[0]*phi.
    For real PDE logic, you'd do partial derivatives along x,y,z dimensions.
    """
    dphi = np.gradient(phi, dx, axis=0)
    gauge_term = -1j * e_charge * A[0] * phi
    return dphi + gauge_term

def gauge_potential_sq(A):
    """
    Sum of squares of A[mu], i.e. A^2 = sum_mu (A[mu]^2).
    shape of A: (4, Nx,Ny,Nz).
    """
    return np.sum(A**2, axis=0)

def field_strength(A, dx):
    """
    Compute F[mu,nu] = partial_mu(A[nu]) - partial_nu(A[mu]).
    shape(A) = (4, Nx,Ny,Nz).
    We'll do partial derivatives w.r.t. axes. This is a placeholder approach.
    """
    shape = A.shape
    F = np.zeros((4,4)+shape[1:], dtype=A.dtype)

    def partial_derivs(Acomp):
        dims = Acomp.ndim
        grads = []
        for ax in range(dims):
            grads.append(np.gradient(Acomp, dx, axis=ax))
        return grads

    for mu in range(4):
        for nu in range(mu+1,4):
            A_nu = A[nu]
            A_mu = A[mu]
            grads_nu = partial_derivs(A_nu)
            grads_mu = partial_derivs(A_mu)
            partial_mu_nu = 0
            partial_nu_mu = 0
            if len(grads_nu)>mu:
                partial_mu_nu = grads_nu[mu]
            if len(grads_mu)>nu:
                partial_nu_mu = grads_mu[nu]
            F[mu,nu] = partial_mu_nu - partial_nu_mu
            F[nu,mu] = -(partial_mu_nu - partial_nu_mu)
    return F

###############################################################################
# 2. Potential & PDE Operators
###############################################################################

def potential_derivative(phi, phi_other, lambda_, v, delta):
    """
    dV/dphi for a potential ~ lambda_*(phi^2 - v^2)*phi + delta*(phi - phi_other)
    You can add phi^6, gradient coupling, etc. as needed.
    """
    return lambda_*(phi**2 - v**2)*phi + delta*(phi - phi_other)

def covariant_laplacian(phi, A, e_charge, dx):
    """
    Covariant wave operator D^mu D_mu on phi: 
    naive combination of finite_difference_laplacian + gauge_derivative.
    """
    lap = finite_difference_laplacian(phi, dx)
    gterm = gauge_derivative(phi, A, e_charge, dx)
    return lap + gterm

def PDE_operator(phi, phi_other, A, e_charge, dx, lambda_, v, delta):
    """
    PDE operator: D^mu D_mu phi - dV/dphi
    for a single field phi with partner phi_other.
    """
    lap = covariant_laplacian(phi, A, e_charge, dx)
    dV  = potential_derivative(phi, phi_other, lambda_, v, delta)
    return lap - dV

###############################################################################
# 3. Leapfrog Step (Fixed Step)
###############################################################################

def leapfrog_step(phi1, pi1, phi2, pi2, A, dt, dx, params):
    """
    Single leapfrog step for two scalar fields (phi1, phi2) with
    conjugate momenta (pi1, pi2), plus gauge field A.

    PDE_operator(...) returns the second derivative operator for each phi.
    We'll do half-step in pi, full-step in phi, etc.
    """
    e_charge = params["e_gauge"]
    lambda_e = params["lambda_e"]
    v_e      = params["v_e"]
    delta_e  = params["delta_e"]

    # half-step momenta
    pi1_half = pi1 + 0.5*dt * PDE_operator(phi1, phi2, A, e_charge, dx, lambda_e, v_e, delta_e)
    pi2_half = pi2 + 0.5*dt * PDE_operator(phi2, phi1, A, e_charge, dx, lambda_e, v_e, delta_e)

    # full-step fields
    phi1_new = phi1 + dt*pi1_half
    phi2_new = phi2 + dt*pi2_half

    # gauge update placeholder (real code: update E,B or store A_mom)
    A_new = A  # skip for now

    # second half-step momenta
    pi1_new = pi1_half + 0.5*dt * PDE_operator(phi1_new, phi2_new, A_new, e_charge, dx, lambda_e, v_e, delta_e)
    pi2_new = pi2_half + 0.5*dt * PDE_operator(phi2_new, phi1_new, A_new, e_charge, dx, lambda_e, v_e, delta_e)

    return phi1_new, pi1_new, phi2_new, pi2_new, A_new

def evolve_fields(phi1_init, phi2_init, A_init, tau_end, dt, dx, params):
    """
    Fixed-step PDE evolution from t=0..tau_end using leapfrog_step.
    """
    shape = phi1_init.shape
    pi1 = np.zeros(shape, dtype=phi1_init.dtype)
    pi2 = np.zeros(shape, dtype=phi2_init.dtype)

    phi1 = phi1_init.copy()
    phi2 = phi2_init.copy()
    A    = A_init.copy()

    tau  = 0.0
    out  = []

    while tau < tau_end:
        out.append((tau, phi1.copy(), phi2.copy(), A.copy()))
        if tau + dt > tau_end:
            dt = tau_end - tau
        phi1, pi1, phi2, pi2, A = leapfrog_step(phi1, pi1, phi2, pi2, A, dt, dx, params)
        tau += dt

    out.append((tau, phi1.copy(), phi2.copy(), A.copy()))
    return out

###############################################################################
# 4. Local Error Norm & Adaptive Evolution
###############################################################################

def local_error_norm(phi1_big, pi1_big, phi2_big, pi2_big, A_big,
                     phi1_small, pi1_small, phi2_small, pi2_small, A_small):
    """
    Compare final states from one big step vs. two half steps.
    Return a scalar measure of difference.
    """
    diff_phi1 = phi1_big - phi1_small
    diff_phi2 = phi2_big - phi2_small
    diff_A    = A_big    - A_small

    err_phi1 = np.sqrt(np.sum(np.abs(diff_phi1)**2))
    err_phi2 = np.sqrt(np.sum(np.abs(diff_phi2)**2))
    err_A    = np.sqrt(np.sum(np.abs(diff_A)**2))

    return err_phi1 + err_phi2 + err_A

def evolve_fields_adaptive(phi1_init, phi2_init, A_init, tau_end, dt_initial, dx, params,
                           err_tolerance=1e-3, dt_min=1e-12, dt_max=0.1):
    """
    Adaptive PDE evolution using a "two half-steps vs. one full-step" approach for error estimation.
    We store pi1, pi2 as momenta. The code tries each step with dt:
      1) One big step
      2) Two half steps
    Compare final states => local error
    If error > err_tolerance => reduce dt, retry
    If error < err_tolerance => accept step
    Possibly enlarge dt if error is much smaller than tolerance.

    :return: list of (tau, phi1, phi2, A)
    """
    shape = phi1_init.shape
    pi1 = np.zeros(shape, dtype=phi1_init.dtype)
    pi2 = np.zeros(shape, dtype=phi2_init.dtype)

    phi1 = phi1_init.copy()
    phi2 = phi2_init.copy()
    A    = A_init.copy()

    tau = 0.0
    dt  = dt_initial
    out = []

    while tau < tau_end:
        out.append((tau, phi1.copy(), phi2.copy(), A.copy()))

        if tau + dt > tau_end:
            dt = tau_end - tau
            if dt < dt_min:
                print("Adaptive solver: dt_min exceeded near end, stopping.")
                break

        # One big step
        phi1_big, pi1_big, phi2_big, pi2_big, A_big = leapfrog_step(phi1, pi1, phi2, pi2, A, dt, dx, params)

        # Two half steps
        half_dt = 0.5*dt
        # first half
        phi1_half, pi1_half, phi2_half, pi2_half, A_half = leapfrog_step(phi1, pi1, phi2, pi2, A, half_dt, dx, params)
        # second half
        phi1_small, pi1_small, phi2_small, pi2_small, A_small = leapfrog_step(phi1_half, pi1_half, phi2_half, pi2_half, A_half, half_dt, dx, params)

        # local error
        err = local_error_norm(phi1_big, pi1_big, phi2_big, pi2_big, A_big,
                               phi1_small, pi1_small, phi2_small, pi2_small, A_small)

        if not np.isfinite(err):
            dt *= 0.5
            if dt < dt_min:
                print("Adaptive solver: dt_min exceeded due to overflow, stopping.")
                break
            continue  # retry
        elif err > err_tolerance:
            dt *= 0.5
            if dt < dt_min:
                print("Adaptive solver: dt_min exceeded, stopping.")
                break
            continue  # retry
        else:
            # Accept the two-half-step final
            phi1, pi1, phi2, pi2, A = phi1_small, pi1_small, phi2_small, pi2_small, A_small
            tau += dt

            # optional dt growth
            if err < 0.1*err_tolerance and dt < dt_max:
                dt *= 1.2

    out.append((tau, phi1.copy(), phi2.copy(), A.copy()))
    return out
