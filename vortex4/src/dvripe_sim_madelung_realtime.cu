/******************************************************************************
 * dvripe_sim_madelung_realtime.cu
 *
 * A single-file CUDA + OpenGL + GLFW program that treats the DVRIPE field
 * "like a fluid" in 2D using the Madelung transform:
 *
 *   ψ = sqrt(ρ) * exp(i θ).
 *
 * Then:
 *   ρ behaves like fluid density,
 *   θ's gradient behaves like velocity,
 *   PDE includes a "quantum pressure" term.
 *
 * We do a minimal real-time visualization of the density ρ (grayscale).
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_madelung_realtime.cu -o dvripe_sim_madelung_realtime.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lglew32 -lglfw3 -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_madelung_realtime.cu -o dvripe_sim_madelung_realtime \
 *       -lGLEW -lglfw -lGL
 *
 * Caveats:
 *   - Explicit Euler can be unstable if dt is large or if focusing is strong.
 *   - We show only 2D, periodic boundary conditions.
 *   - For real DVRIPE research, more advanced integrators / 3D might be needed.
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// For M_PI on MSVC
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Prevent GLFW from including OpenGL headers
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Include GLEW after GLFW
#include <GL/glew.h>

#include <cuda_gl_interop.h>

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------
static const int    NX = 256;      // grid size in x
static const int    NY = 256;      // grid size in y
static const float  DX = 0.1f;     // spatial resolution
static const float  DT = 0.00001f;  // time step
static const float  G  = 0.1f;     // focusing (nonlinear) strength
static const int    STEPS_PER_FRAME = 1; // PDE steps per frame

// We'll store two 2D arrays on the GPU: rho (density) and theta (phase).
// The velocity is derived from theta, and quantum pressure from rho.

// -----------------------------------------------------------------------------
// Device utility functions
// -----------------------------------------------------------------------------
__device__ inline float periodicCoord(int i, int n)
{
    // wrap i into [0..n-1]
    if(i < 0)   i += n;
    if(i >= n)  i -= n;
    return float(i);
}

__device__ inline int periodicIndex(int i, int n)
{
    if(i < 0)   i += n;
    if(i >= n)  i -= n;
    return i;
}

// -----------------------------------------------------------------------------
// We'll define a kernel that updates rho and theta in two steps each iteration:
//   1) Continuity: ∂t ρ = -∇ · (ρ v)
//   2) Phase eq:  ∂t θ = - (1/2)|v|^2 - G ρ + quantumPressure
//
// with v = ∇θ, quantumPressure ~ - (1 / 2 sqrt(ρ)) ∇² sqrt(ρ).
// We'll do an explicit Euler step, with periodic BC.
// -----------------------------------------------------------------------------

// We'll store intermediate arrays in global memory for v_x, v_y, Q, sqrtR, lapSqrtR, etc.

__global__ void madelungUpdateKernel(float* rho, float* theta,
                                     float* rho_new, float* theta_new,
                                     float* v_x, float* v_y,
                                     float* sqrtR, float* lapSqrtR,
                                     float* quantumP,
                                     int nx, int ny,
                                     float dx, float dt, float g)
{
    // 1) Compute velocity v = ∇θ
    // 2) Compute sqrt(ρ) and its Laplacian => quantum pressure
    // 3) Update ρ via continuity
    // 4) Update θ

    // We'll do this in multiple passes or inlined for clarity.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= nx || j >= ny) return;

    int idx = j*nx + i;

    // 1) velocity v = ∇θ
    // central difference with periodic BC
    int ip = periodicIndex(i+1, nx);
    int im = periodicIndex(i-1, nx);
    int jp = periodicIndex(j+1, ny);
    int jm = periodicIndex(j-1, ny);

    float theta_c = theta[idx];
    float theta_ip = theta[j*nx + ip];
    float theta_im = theta[j*nx + im];
    float theta_jp = theta[jp*nx + i];
    float theta_jm = theta[jm*nx + i];

    // We'll do a simple central difference
    float dtheta_dx = (theta_ip - theta_im)/(2.0f*dx);
    float dtheta_dy = (theta_jp - theta_jm)/(2.0f*dx); // assume same dx in y

    // Because phase is in [-π, π], we might want to unwrap. But let's keep it simple here.
    v_x[idx] = dtheta_dx;
    v_y[idx] = dtheta_dy;

    // 2) sqrt(ρ) and its Laplacian => quantum pressure
    float r_c = rho[idx];
    float sr_c = sqrtf(r_c);
    sqrtR[idx] = sr_c;

    // We'll compute laplacian in the same kernel but we need neighbors of sr
    // We'll finish this after we do a __syncthreads() at the end of this pass
}

// Second pass: compute laplacian of sqrtR, then quantumP, then do continuity & phase
__global__ void madelungFinishKernel(float* rho, float* theta,
                                     float* rho_new, float* theta_new,
                                     float* v_x, float* v_y,
                                     float* sqrtR, float* lapSqrtR,
                                     float* quantumP,
                                     int nx, int ny,
                                     float dx, float dt, float g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i >= nx || j >= ny) return;

    int idx = j*nx + i;

    int ip = periodicIndex(i+1, nx);
    int im = periodicIndex(i-1, nx);
    int jp = periodicIndex(j+1, ny);
    int jm = periodicIndex(j-1, ny);

    float srC = sqrtR[idx];
    float srIp = sqrtR[j*nx + ip];
    float srIm = sqrtR[j*nx + im];
    float srJp = sqrtR[jp*nx + i];
    float srJm = sqrtR[jm*nx + i];

    // Laplacian(sqrt(rho)) with central differences
    float lap = (srIp + srIm + srJp + srJm - 4.0f*srC)/(dx*dx);
    lapSqrtR[idx] = lap;

    // quantum pressure Q = - (1 / 2 sqrt(rho)) * laplacian(sqrt(rho))
    // We'll check for srC > 1e-9 to avoid /0
    float Q = 0.0f;
    if(srC > 1e-9f) {
        Q = -0.5f * lap / srC;
    }
    quantumP[idx] = Q;

    // 3) Update ρ: ∂t ρ = -∇·(ρ v)
    // ∇·(ρ v) ~ d/dx(ρ v_x) + d/dy(ρ v_y)
    float rho_c = rho[idx];
    float vx_c = v_x[idx];
    float vy_c = v_y[idx];

    float rho_ip = rho[j*nx + ip];
    float vx_ip  = v_x[j*nx + ip];
    float rho_im = rho[j*nx + im];
    float vx_im  = v_x[j*nx + im];
    float rho_jp = rho[jp*nx + i];
    float vy_jp  = v_y[jp*nx + i];
    float rho_jm = rho[jm*nx + i];
    float vy_jm  = v_y[jm*nx + i];

    float flux_xp = rho_ip*vx_ip;
    float flux_xm = rho_im*vx_im;
    float flux_yp = rho_jp*vy_jp;
    float flux_ym = rho_jm*vy_jm;

    // central difference for div
    float div = ((flux_xp - flux_xm) + (flux_yp - flux_ym)) / (2.0f*dx);
    float rho_new_val = rho_c - dt*div;
    if(rho_new_val < 0.0f) rho_new_val = 0.0f; // clamp
    rho_new[idx] = rho_new_val;

    // 4) Update θ: ∂t θ = - (1/2)|v|^2 - g ρ + Q
    float vx2 = vx_c*vx_c;
    float vy2 = vy_c*vy_c;
    float kinetic = 0.5f*(vx2 + vy2);
    float dtheta_dt = - (kinetic + g*rho_c - Q);
    float theta_new_val = theta[idx] + dt*dtheta_dt;

    // keep phase in [-π, π] for neatness (optional)
    // not strictly needed, but can help avoid big jumps
    if(theta_new_val >  M_PI) theta_new_val -= 2.0f*(float)M_PI;
    if(theta_new_val < -M_PI) theta_new_val += 2.0f*(float)M_PI;

    theta_new[idx] = theta_new_val;
}

// A kernel to copy rho_new -> rho, theta_new -> theta
__global__ void copyFields(float* rho, float* rho_new,
                           float* theta, float* theta_new,
                           int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        rho[idx] = rho_new[idx];
        theta[idx] = theta_new[idx];
    }
}

// -----------------------------------------------------------------------------
// Host function to initialize the field
// We'll define a big Gaussian amplitude + possibly a swirl in θ
// -----------------------------------------------------------------------------
void initializeMadelung(std::vector<float>& rhoHost,
                        std::vector<float>& thetaHost,
                        int nx, int ny, float dx)
{
    float cx = nx/2.0f;
    float cy = ny/2.0f;

    for(int j=0; j<ny; j++){
        for(int i=0; i<nx; i++){
            int idx = j*nx + i;
            float x = i - cx;
            float y = j - cy;
            float r2 = x*x + y*y;

            // amplitude => Gaussian
            float amp = std::exp(-r2 / (nx*dx*0.3f));
            rhoHost[idx] = amp*amp; // or amp^2

            // Optionally swirl the phase:
            float theta = std::atan2(y, x); // single vortex
            thetaHost[idx] = theta;
        }
    }
}

// -----------------------------------------------------------------------------
// We'll display the density ρ in grayscale
// -----------------------------------------------------------------------------
__global__ void fillPBOKernel(uchar4* pbo, const float* rho, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < nx && j < ny)
    {
        int idx = j*nx + i;
        float val = rho[idx]*10.0f; // arbitrary scale
        if(val>1.0f) val=1.0f;
        unsigned char c = (unsigned char)(val*255.0f);

        pbo[idx].x = c;
        pbo[idx].y = c;
        pbo[idx].z = c;
        pbo[idx].w = 255;
    }
}

// -----------------------------------------------------------------------------
// CUDA-OpenGL Interop
// -----------------------------------------------------------------------------
static GLuint pboID;
static struct cudaGraphicsResource* pboResource;

bool createPBO(int width, int height)
{
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&pboResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard);
    if(err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

void renderPBO(int width, int height)
{
    glRasterPos2i(0,0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(NX, NY, "DVRIPE - Madelung Fluid 2D", nullptr, nullptr);
    if(!window){
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 2) Init GLEW
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK){
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 3) Allocate host memory
    int n = NX*NY;
    std::vector<float> rhoHost(n), thetaHost(n);
    initializeMadelung(rhoHost, thetaHost, NX, NY, DX);

    // 4) Allocate device memory
    float *rhoDev, *thetaDev, *rhoNewDev, *thetaNewDev;
    float *vXDev, *vYDev, *sqrtRDev, *lapSqrtRDev, *qPressDev;
    cudaMalloc(&rhoDev,     n*sizeof(float));
    cudaMalloc(&thetaDev,   n*sizeof(float));
    cudaMalloc(&rhoNewDev,  n*sizeof(float));
    cudaMalloc(&thetaNewDev,n*sizeof(float));
    cudaMalloc(&vXDev,      n*sizeof(float));
    cudaMalloc(&vYDev,      n*sizeof(float));
    cudaMalloc(&sqrtRDev,   n*sizeof(float));
    cudaMalloc(&lapSqrtRDev,n*sizeof(float));
    cudaMalloc(&qPressDev,  n*sizeof(float));

    cudaMemcpy(rhoDev,   rhoHost.data(),   n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(thetaDev, thetaHost.data(), n*sizeof(float), cudaMemcpyHostToDevice);

    // 5) Create PBO for real-time display of ρ
    if(!createPBO(NX, NY)){
        std::cerr << "Failed to create PBO\n";
        return -1;
    }

    // 6) CUDA config
    dim3 threads(16,16);
    dim3 blocks( (NX+threads.x-1)/threads.x,
                 (NY+threads.y-1)/threads.y );

    // copy kernel config
    int copyThreads=256;
    int copyBlocks=(n+copyThreads-1)/copyThreads;

    // 7) Main loop
    while(!glfwWindowShouldClose(window))
    {
        // PDE steps
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // pass 1: compute velocity + sqrt(rho)
            madelungUpdateKernel<<<blocks, threads>>>(rhoDev, thetaDev,
                                                      rhoNewDev, thetaNewDev,
                                                      vXDev, vYDev,
                                                      sqrtRDev, lapSqrtRDev, qPressDev,
                                                      NX, NY, DX, DT, G);
            cudaDeviceSynchronize();

            // pass 2: finish quantum pressure, update continuity & phase
            madelungFinishKernel<<<blocks, threads>>>(rhoDev, thetaDev,
                                                      rhoNewDev, thetaNewDev,
                                                      vXDev, vYDev,
                                                      sqrtRDev, lapSqrtRDev, qPressDev,
                                                      NX, NY, DX, DT, G);
            cudaDeviceSynchronize();

            // copy fields back
            copyFields<<<copyBlocks,  copyThreads>>>(rhoDev, rhoNewDev, thetaDev, thetaNewDev, n);
            cudaDeviceSynchronize();
        }

        // Map PBO and fill with ρ
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        uchar4* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillPBOKernel<<<blocks, threads>>>(d_pbo, rhoDev, NX, NY);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &pboResource, 0);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0,0,NX,NY);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0,NX,0,NY,-1,1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        renderPBO(NX, NY);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);

    cudaFree(rhoDev);
    cudaFree(thetaDev);
    cudaFree(rhoNewDev);
    cudaFree(thetaNewDev);
    cudaFree(vXDev);
    cudaFree(vYDev);
    cudaFree(sqrtRDev);
    cudaFree(lapSqrtRDev);
    cudaFree(qPressDev);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
