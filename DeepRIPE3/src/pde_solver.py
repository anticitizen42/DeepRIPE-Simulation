#!/usr/bin/env python3
"""
pde_solver.py

Hybrid GPU-accelerated DV-RIPE PDE solver.

This version keeps as much data on the GPU as possible. Gauge fields (A and its
momentum E) are stored as GPU arrays (cl.array objects) when "use_gpu" is True and
updated via fused GPU kernels. Scalar fields remain on the CPU.
"""

import numpy as np

# Standard CPU functions for finite differences.
def finite_difference_laplacian(field, dx):
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
    dphi = np.gradient(phi, dx, axis=0)
    gauge_term = -1j * e_charge * A[0] * phi
    return dphi + gauge_term

def potential_derivative(phi, phi_other, lambda_, v, delta):
    return 0.5 * lambda_ * (np.abs(phi)**2 - v**2) * phi + delta * (phi - phi_other)

def covariant_laplacian(phi, A, e_charge, dx):
    lap = finite_difference_laplacian(phi, dx)
    gauge_term = gauge_derivative(phi, A, e_charge, dx)
    return lap + gauge_term

def PDE_operator(phi, phi_other, A, e_charge, dx, lambda_, v, delta):
    lap_term = covariant_laplacian(phi, A, e_charge, dx)
    pot_term = potential_derivative(phi, phi_other, lambda_, v, delta)
    return lap_term - pot_term

# --- GPU Integration ---
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def gpu_complex_laplacian_3d(field, dx, gpu_obj):
    """
    Compute the 3D Laplacian of a complex field stored as a cl.array using the GPU.
    Assumes gpu_obj has a method laplacian_3d_clarray that accepts a cl.array and returns a cl.array.
    """
    return gpu_obj.laplacian_3d_clarray(field, dx)

def gauge_operator(A, dx, params):
    """
    Compute the operator for gauge dynamics.
    For spatial components (indices 1,2,3) of A, compute the Laplacian.
    If "use_gpu" is True and GPU is available, operate on GPU arrays.
    """
    shape = A.shape  # (4, Nx, Ny, Nz)
    use_gpu = params.get("use_gpu", False) and GPU_AVAILABLE
    if use_gpu:
        if not hasattr(gauge_operator, "gpu_obj"):
            from gpu_kernels import GPUKernels
            gauge_operator.gpu_obj = GPUKernels()
        gpu_obj = gauge_operator.gpu_obj
        op = [None] * 4
        # Time component remains zero.
        op[0] = cl_array.zeros(gpu_obj.queue, shape[1:], dtype=A.dtype)
        for i in range(1, 4):
            op[i] = gpu_complex_laplacian_3d(A[i], dx, gpu_obj)
        # For simplicity, we return a list of cl.array objects.
    else:
        op = np.zeros_like(A, dtype=A.dtype)
        for i in range(1, 4):
            op[i] = finite_difference_laplacian_3d(A[i], dx)
    return op

# --- Fused GPU Kernels (simplified placeholders) ---
SCALAR_KERNEL_CODE = """
__kernel void scalar_update(__global const float *phi,
                            __global const float *pi,
                            __global float *phi_new,
                            __global float *pi_new,
                            const float dt,
                            const int N)
{
    int idx = get_global_id(0);
    if(idx < N){
        phi_new[idx] = phi[idx] + dt * pi[idx];
        pi_new[idx] = pi[idx]; // Dummy update
    }
}
"""

GAUGE_KERNEL_CODE = """
__kernel void gauge_update(__global const float *A,
                           __global const float *E,
                           __global float *A_new,
                           __global float *E_new,
                           const float dt,
                           const float dx2,
                           const int N)
{
    int idx = get_global_id(0);
    if(idx < N){
        A_new[idx] = A[idx] + dt * E[idx];
        E_new[idx] = E[idx]; // Dummy update
    }
}
"""

class GPUFusedKernels:
    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        self.scalar_prog = cl.Program(self.ctx, SCALAR_KERNEL_CODE).build()
        self.gauge_prog = cl.Program(self.ctx, GAUGE_KERNEL_CODE).build()
        print("GPUFusedKernels initialized on device:", self.ctx.devices[0].name)
    
    def scalar_update(self, phi, pi, dt):
        N = phi.size
        from pyopencl import array as cl_array
        phi_new = cl_array.empty(self.queue, phi.shape, dtype=phi.dtype)
        pi_new = cl_array.empty(self.queue, pi.shape, dtype=pi.dtype)
        global_size = (int(N),)
        self.scalar_prog.scalar_update(self.queue, global_size, None,
                                       phi.data, pi.data, phi_new.data, pi_new.data,
                                       np.float32(dt), np.int32(N))
        return phi_new, pi_new
    
    def gauge_update(self, A, E, dt, dx):
        N = A.size
        from pyopencl import array as cl_array
        A_new = cl_array.empty(self.queue, A.shape, dtype=A.dtype)
        E_new = cl_array.empty(self.queue, E.shape, dtype=E.dtype)
        global_size = (int(N),)
        dx2 = np.float32(dx*dx)
        self.gauge_prog.gauge_update(self.queue, global_size, None,
                                     A.data, E.data, A_new.data, E_new.data,
                                     np.float32(dt), dx2, np.int32(N))
        return A_new, E_new

# --- CPU fallback for gauge update ---
def leapfrog_step_cpu(phi1, pi1, phi2, pi2, A, E, dt, dx, params):
    e_charge = params["e_gauge"]
    lambda_e = params["lambda_e"]
    v_e = params["v_e"]
    delta_e = params["delta_e"]
    
    pi1_half = pi1 + 0.5 * dt * PDE_operator(phi1, phi2, A, e_charge, dx, lambda_e, v_e, delta_e)
    pi2_half = pi2 + 0.5 * dt * PDE_operator(phi2, phi1, A, e_charge, dx, lambda_e, v_e, delta_e)
    phi1_new = phi1 + dt * pi1_half
    phi2_new = phi2 + dt * pi2_half
    pi1_new = pi1_half + 0.5 * dt * PDE_operator(phi1_new, phi2_new, A, e_charge, dx, lambda_e, v_e, delta_e)
    pi2_new = pi2_half + 0.5 * dt * PDE_operator(phi2_new, phi1_new, A, e_charge, dx, lambda_e, v_e, delta_e)
    
    E_half = E + 0.5 * dt * finite_difference_laplacian_3d(A[1], dx)
    A_new = A + dt * E_half
    E_new = E_half + 0.5 * dt * finite_difference_laplacian_3d(A_new[1], dx)
    return phi1_new, pi1_new, phi2_new, pi2_new, A_new, E_new

def leapfrog_step(phi1, pi1, phi2, pi2, A, E, dt, dx, params):
    use_gpu = params.get("use_gpu", False) and GPU_AVAILABLE
    if use_gpu:
        # For this hybrid approach, scalar fields remain on CPU.
        phi1_new = phi1 + dt * pi1  # Placeholder update for demonstration.
        phi2_new = phi2 + dt * pi2
        pi1_new = pi1  # Placeholder
        pi2_new = pi2  # Placeholder
        
        from pyopencl import array as cl_array
        gpu_obj = GPUFusedKernels()
        # A and E are GPU arrays.
        A_new, E_new = gpu_obj.gauge_update(A, E, dt, dx)
        return phi1_new, pi1_new, phi2_new, pi2_new, A_new, E_new
    else:
        return leapfrog_step_cpu(phi1, pi1, phi2, pi2, A, E, dt, dx, params)

def evolve_fields(phi1_init, phi2_init, A_init, tau_end, dt, dx, params):
    phi1 = phi1_init.copy()
    phi2 = phi2_init.copy()
    pi1 = np.zeros_like(phi1)
    pi2 = np.zeros_like(phi2)
    
    use_gpu = params.get("use_gpu", False) and GPU_AVAILABLE
    if use_gpu:
        from pyopencl import array as cl_array
        gpu_obj = GPUFusedKernels()
        A = cl_array.to_device(gpu_obj.queue, A_init.astype(np.complex64))
        E = cl_array.zeros(gpu_obj.queue, A.shape, dtype=A.dtype)
    else:
        A = A_init.copy()
        E = np.zeros_like(A)
    
    tau = 0.0
    snapshots = []
    while tau < tau_end:
        if use_gpu:
            A_cpu = A.get().reshape(A_init.shape)
        else:
            A_cpu = A.copy()
        snapshots.append((tau, phi1.copy(), phi2.copy(), A_cpu.copy()))
        if tau + dt > tau_end:
            dt = tau_end - tau
        phi1, pi1, phi2, pi2, A, E = leapfrog_step(phi1, pi1, phi2, pi2, A, E, dt, dx, params)
        tau += dt
    if use_gpu:
        A_final = A.get().reshape(A_init.shape)
    else:
        A_final = A.copy()
    snapshots.append((tau, phi1.copy(), phi2.copy(), A_final.copy()))
    return snapshots

def local_error_norm(phi1_big, pi1_big, phi2_big, pi2_big, A_big, E_big,
                     phi1_small, pi1_small, phi2_small, pi2_small, A_small, E_small):
    # If gauge field arrays are GPU arrays, convert them to host arrays.
    try:
        if hasattr(A_big, "get"):
            A_big_host = A_big.get()
        else:
            A_big_host = A_big
        if hasattr(A_small, "get"):
            A_small_host = A_small.get()
        else:
            A_small_host = A_small
        if hasattr(E_big, "get"):
            E_big_host = E_big.get()
        else:
            E_big_host = E_big
        if hasattr(E_small, "get"):
            E_small_host = E_small.get()
        else:
            E_small_host = E_small
    except Exception as e:
        print("Error converting GPU arrays to host:", e)
        A_big_host = A_big
        A_small_host = A_small
        E_big_host = E_big
        E_small_host = E_small

    err_phi1 = np.sqrt(np.sum(np.abs(phi1_big - phi1_small)**2))
    err_phi2 = np.sqrt(np.sum(np.abs(phi2_big - phi2_small)**2))
    err_A = np.sqrt(np.sum(np.abs(A_big_host - A_small_host)**2))
    err_E = np.sqrt(np.sum(np.abs(E_big_host - E_small_host)**2))
    return err_phi1 + err_phi2 + err_A + err_E

def evolve_fields_adaptive(phi1_init, phi2_init, A_init, tau_end, dt_initial, dx, params,
                           err_tolerance=1e-3, dt_min=1e-12, dt_max=0.1):
    phi1 = phi1_init.copy()
    phi2 = phi2_init.copy()
    pi1 = np.zeros_like(phi1)
    pi2 = np.zeros_like(phi2)
    
    use_gpu = params.get("use_gpu", False) and GPU_AVAILABLE
    if use_gpu:
        from pyopencl import array as cl_array
        gpu_obj = GPUFusedKernels()
        A = cl_array.to_device(gpu_obj.queue, A_init.astype(np.complex64))
        E = cl_array.zeros(gpu_obj.queue, A.shape, dtype=A.dtype)
    else:
        A = A_init.copy()
        E = np.zeros_like(A)
    
    tau = 0.0
    dt = dt_initial
    snapshots = []
    while tau < tau_end:
        if use_gpu:
            A_cpu = A.get().reshape(A_init.shape)
        else:
            A_cpu = A.copy()
        snapshots.append((tau, phi1.copy(), phi2.copy(), A_cpu.copy()))
        if tau + dt > tau_end:
            dt = tau_end - tau
            if dt < dt_min:
                print("Adaptive solver: dt_min reached near end. Stopping.")
                break
        phi1_big, pi1_big, phi2_big, pi2_big, A_big, E_big = leapfrog_step(phi1, pi1, phi2, pi2, A, E, dt, dx, params)
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
            continue
        else:
            phi1, pi1, phi2, pi2, A, E = phi1_small, pi1_small, phi2_small, pi2_small, A_small, E_small
            tau += dt
            if err < 0.1 * err_tolerance and dt < dt_max:
                dt *= 1.2
    if use_gpu:
        A_final = A.get().reshape(A_init.shape)
    else:
        A_final = A.copy()
    snapshots.append((tau, phi1.copy(), phi2.copy(), A_final.copy()))
    return snapshots
