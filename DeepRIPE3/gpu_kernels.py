# gpu_kernels.py

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np

# OpenCL kernel code for computing the 3D Laplacian.
KERNEL_CODE = """
__kernel void laplacian_3d(__global const float *field,
                           __global float *lap,
                           const int Nx, const int Ny, const int Nz,
                           const float dx2)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    
    // Avoid boundaries.
    if(i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && k > 0 && k < Nz-1)
    {
        int idx = i * Ny * Nz + j * Nz + k;
        int idx_ip = (i+1) * Ny * Nz + j * Nz + k;
        int idx_im = (i-1) * Ny * Nz + j * Nz + k;
        int idx_jp = i * Ny * Nz + (j+1) * Nz + k;
        int idx_jm = i * Ny * Nz + (j-1) * Nz + k;
        int idx_kp = i * Ny * Nz + j * Nz + (k+1);
        int idx_km = i * Ny * Nz + j * Nz + (k-1);
        
        lap[idx] = (field[idx_ip] + field[idx_im] +
                    field[idx_jp] + field[idx_jm] +
                    field[idx_kp] + field[idx_km] -
                    6.0f * field[idx]) / dx2;
    }
}
"""

class GPUKernels:
    def __init__(self):
        try:
            # Create a context on the first available GPU.
            self.ctx = cl.create_some_context(interactive=False)
            self.queue = cl.CommandQueue(self.ctx)
            self.program = cl.Program(self.ctx, KERNEL_CODE).build()
            print("GPUKernels initialized. Using device:", self.ctx.devices[0].name)
        except Exception as e:
            print("Error initializing GPUKernels:", e)
            raise

    def laplacian_3d_clarray(self, field_cl, dx):
        """
        Compute the 3D Laplacian of a field stored as a cl.array.
        This function now first copies the input to ensure the array starts at offset zero.
        
        Parameters:
            field_cl: a cl.array of shape (Nx, Ny, Nz) with dtype=np.float32.
            dx: grid spacing (float)
            
        Returns:
            lap_cl: a cl.array containing the Laplacian of the field.
        """
        # Ensure the input array is contiguous from offset zero.
        field_cl = field_cl.copy()  # Remove any nonzero offset.
        
        # Get shape.
        Nx, Ny, Nz = field_cl.shape
        # Allocate output cl.array.
        lap_cl = cl_array.empty(self.queue, field_cl.shape, dtype=field_cl.dtype)
        dx2 = np.float32(dx * dx)
        
        # Set global size.
        global_size = (Nx, Ny, Nz)
        
        # Debug print to show kernel invocation.
        print("Invoking GPU laplacian kernel on grid size:", global_size)
        
        self.program.laplacian_3d(self.queue, global_size, None,
                                  field_cl.data, lap_cl.data,
                                  np.int32(Nx), np.int32(Ny), np.int32(Nz), dx2)
        self.queue.finish()
        return lap_cl

# Quick test: run this module directly.
if __name__ == "__main__":
    Nx, Ny, Nz = 64, 64, 64
    field_np = np.random.rand(Nx, Ny, Nz).astype(np.float32)
    dx = 0.1
    gpu_obj = GPUKernels()
    field_cl = cl_array.to_device(gpu_obj.queue, field_np)
    lap_cl = gpu_obj.laplacian_3d_clarray(field_cl, dx)
    lap_result = lap_cl.get()
    print("Laplacian computed on GPU, shape:", lap_result.shape)
