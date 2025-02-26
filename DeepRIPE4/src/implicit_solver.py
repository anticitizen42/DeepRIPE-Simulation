#!/usr/bin/env python3
"""
src/implicit_solver.py
Version 1.0

This module provides a robust Newton–Krylov solver for solving nonlinear systems F(x)=0,
using a matrix‑free approach. It is designed to handle the implicit systems that arise in
our DV‑RIPE simulation's high‑order integrator.

Key Features:
  - Matrix‑free Jacobian–vector product using finite-difference approximation (if no analytic function is provided).
  - Linear system solve via GMRES from scipy.sparse.linalg.
  - Optional preconditioning support.
  
Usage:
    solution = newton_krylov_solve(F, x0, tol=1e-6, max_iter=50)
"""

import numpy as np
import logging
from scipy.sparse.linalg import gmres

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def newton_krylov_solve(F, x0, tol=1e-6, max_iter=50, jacobian_vector_product=None, preconditioner=None):
    """
    Solve the nonlinear system F(x)=0 using a Newton–Krylov method.
    
    Parameters:
      F: function(x) -> ndarray
         Computes the residual F(x) for a given x.
      x0: ndarray
         Initial guess for the solution.
      tol: float
         Tolerance for convergence.
      max_iter: int
         Maximum number of Newton iterations.
      jacobian_vector_product: Optional function (x, v) -> ndarray
         Computes the product J(x)*v. If not provided, finite-difference approximation is used.
      preconditioner: Optional linear operator or function
         Applies a preconditioner to a vector. Use a callable M(v) that returns an approximation
         to the solution of M^{-1} v.
    
    Returns:
      x: ndarray
         The computed solution.
         
    If the linear system is not solved accurately, a warning is logged.
    """
    x = x0.copy()
    for iter in range(max_iter):
        Fx = F(x)
        norm_Fx = np.linalg.norm(Fx)
        logger.debug(f"Newton iteration {iter}: ||F(x)|| = {norm_Fx:.2e}")
        if norm_Fx < tol:
            logger.info(f"Newton–Krylov converged in {iter} iterations.")
            return x
        # Define Jacobian-vector product function.
        if jacobian_vector_product is None:
            def Jv(v):
                eps = 1e-8
                return (F(x + eps * v) - Fx) / eps
        else:
            def Jv(v):
                return jacobian_vector_product(x, v)
        
        # Define a linear operator A(v) = J(x)*v.
        A = lambda v: Jv(v)
        # Solve the linear system J(x)*delta = -F(x) using GMRES.
        delta, exitCode = gmres(A, -Fx, M=preconditioner, tol=tol)
        if exitCode != 0:
            logger.warning(f"GMRES did not converge at iteration {iter} (exit code {exitCode}).")
        x = x + delta
    logger.error("Newton–Krylov did not converge within the maximum number of iterations.")
    return x

if __name__ == "__main__":
    # Test harness for the Newton–Krylov solver.
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running test harness for Newton–Krylov solver.")
    
    # Example problem: Solve x^2 - 2 = 0, solution should be sqrt(2).
    def F_example(x):
        return x**2 - 2.0
    
    x0 = np.array([1.0])
    solution = newton_krylov_solve(F_example, x0, tol=1e-8, max_iter=20)
    error = np.abs(solution**2 - 2)
    logger.info(f"Computed solution: {solution}, error: {error}")
