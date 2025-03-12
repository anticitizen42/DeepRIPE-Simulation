/*
 * dvripe_sim.cu
 *
 * CUDA simulation of the mass-energy field according to the DVRIPE hypothesis.
 * The field ψ evolves according to:
 *
 *    dψ/dt = i * D * ∇²ψ + i * g * |ψ|² ψ
 *
 * where i is the imaginary unit.
 *
 * This program uses a finite-difference method on a 2D grid with periodic boundaries.
 *
 * Compile with: nvcc -o dvripe_sim dvripe_sim.cu
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define NX 256        // Number of grid points in x
#define NY 256        // Number of grid points in y
#define NT 1000       // Number of time steps
#define DT 0.0001f    // Time step size
#define DX 0.1f       // Spatial resolution

// Simulation parameters will be passed as arguments to the kernel
// They are defined on the host when calling the kernel

// Define complex number as float2 (x = real, y = imaginary)

// Device inline functions for complex arithmetic
__device__ inline float2 complexAdd(const float2 a, const float2 b) {
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__device__ inline float2 complexSub(const float2 a, const float2 b) {
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

__device__ inline float2 complexMul(const float2 a, const float2 b) {
    float2 c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__device__ inline float complexAbs2(const float2 a) {
    return a.x * a.x + a.y * a.y;
}

// Multiply complex number by a scalar
__device__ inline float2 complexScalarMul(const float2 a, const float s) {
    float2 c;
    c.x = a.x * s;
    c.y = a.y * s;
    return c;
}

// Multiply a complex number by i (imaginary unit)
__device__ inline float2 complexMulI(const float2 a) {
    // i * a = (-a.y, a.x)
    float2 c;
    c.x = -a.y;
    c.y = a.x;
    return c;
}

// Kernel to update the field psi using finite differences
__global__ void updateField(float2* psi, float2* psi_new, int nx, int ny, float dt, float dx, float D, float G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index

    if (i < nx && j < ny) {
        int idx = j * nx + i;
        
        // Periodic boundary indices
        int ip = (i + 1) % nx;
        int im = (i - 1 + nx) % nx;
        int jp = (j + 1) % ny;
        int jm = (j - 1 + ny) % ny;
        
        int idx_ip = j * nx + ip;
        int idx_im = j * nx + im;
        int idx_jp = jp * nx + i;
        int idx_jm = jm * nx + i;
        
        // Compute Laplacian using central differences
        float2 laplacian;
        laplacian.x = (psi[idx_ip].x + psi[idx_im].x + psi[idx_jp].x + psi[idx_jm].x - 4.0f * psi[idx].x) / (dx * dx);
        laplacian.y = (psi[idx_ip].y + psi[idx_im].y + psi[idx_jp].y + psi[idx_jm].y - 4.0f * psi[idx].y) / (dx * dx);
        
        // Nonlinear term: i * G * |psi|^2 * psi
        float abs2 = complexAbs2(psi[idx]);
        float2 nonlinear = complexScalarMul(psi[idx], G * abs2);
        nonlinear = complexMulI(nonlinear); // multiply by i
        
        // Linear dispersion term: i * D * laplacian
        float2 linear = complexScalarMul(laplacian, D);
        linear = complexMulI(linear); // multiply by i
        
        // Total derivative dψ/dt
        float2 dpsi_dt = complexAdd(linear, nonlinear);
        
        // Euler integration: psi_new = psi + dt * dψ/dt
        psi_new[idx] = complexAdd(psi[idx], complexScalarMul(dpsi_dt, dt));
    }
}

// Kernel to copy psi_new to psi (for the next time step)
__global__ void copyField(float2* psi, float2* psi_new, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        psi[idx] = psi_new[idx];
    }
}

// Host function to initialize the field with a vortex initial condition
void initializeField(float2* psi_host, int nx, int ny, float dx) {
    // Set vortex center to the middle of the grid
    float cx = nx / 2.0f;
    float cy = ny / 2.0f;
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = j * nx + i;
            float x = i - cx;
            float y = j - cy;
            float r = sqrtf(x*x + y*y);
            float theta = atan2f(y, x);
            
            // Amplitude decays with distance from the center
            float amplitude = expf(-r*r / (nx * dx * 0.1f));
            
            // Phase gives a winding (vortex of charge 1)
            float phase = theta;
            
            psi_host[idx].x = amplitude * cosf(phase);
            psi_host[idx].y = amplitude * sinf(phase);
        }
    }
}

int main() {
    // Total number of grid points
    int n = NX * NY;
    
    // Allocate host memory
    float2* psi_host = new float2[n];
    
    // Initialize the field on host
    initializeField(psi_host, NX, NY, DX);
    
    // Allocate device memory
    float2 *psi_dev, *psi_new_dev;
    cudaMalloc((void**)&psi_dev, n * sizeof(float2));
    cudaMalloc((void**)&psi_new_dev, n * sizeof(float2));
    
    // Copy initial field to device
    cudaMemcpy(psi_dev, psi_host, n * sizeof(float2), cudaMemcpyHostToDevice);
    
    // Define CUDA grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Time stepping loop
    for (int t = 0; t < NT; t++) {
        // Update field: pass D and G as parameters (here both are 1.0f)
        updateField<<<numBlocks, threadsPerBlock>>>(psi_dev, psi_new_dev, NX, NY, DT, DX, 1.0f, 1.0f);
        cudaDeviceSynchronize();
        
        // Copy psi_new back to psi for the next iteration
        int numThreads = 256;
        int numBlocks1 = (n + numThreads - 1) / numThreads;
        copyField<<<numBlocks1, numThreads>>>(psi_dev, psi_new_dev, n);
        cudaDeviceSynchronize();
        
        // Optionally, print progress every 100 steps
        if (t % 100 == 0) {
            std::cout << "Completed time step " << t << " / " << NT << std::endl;
        }
    }
    
    // Copy final field back to host
    cudaMemcpy(psi_host, psi_dev, n * sizeof(float2), cudaMemcpyDeviceToHost);
    
    // Write output to file (each grid point: amplitude and phase)
    std::ofstream outfile("output.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file!" << std::endl;
        return 1;
    }
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int idx = j * NX + i;
            float real = psi_host[idx].x;
            float imag = psi_host[idx].y;
            float amplitude = sqrtf(real * real + imag * imag);
            float phase = atan2f(imag, real);
            outfile << amplitude << " " << phase << " ";
        }
        outfile << std::endl;
    }
    outfile.close();
    
    // Clean up
    cudaFree(psi_dev);
    cudaFree(psi_new_dev);
    delete[] psi_host;
    
    std::cout << "Simulation complete. Output written to output.txt" << std::endl;
    return 0;
}
