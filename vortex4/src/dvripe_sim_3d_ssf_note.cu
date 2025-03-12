/******************************************************************************
 * dvripe_sim_3d_ssf_note.cu
 *
 * Single-file CUDA + cuFFT + OpenGL + GLFW program demonstrating a 3D 
 * Split-Step Fourier (SSF) simulation of the focusing Schrödinger PDE:
 *
 *    i ∂ψ/∂t = -D ∇² ψ - G |ψ|² ψ + V_note(r) ψ
 *
 * where V_note(r) is a "Note" potential that can act like a harmonic or 
 * pattern-based trap, stabilizing the vortex nexus in a particular shape.
 *
 * We'll render a single slice (z=NZ/2) in grayscale, and optionally do 
 * swirl detection if desired. For brevity, we only show amplitude in the slice.
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_ssf_note.cu -o dvripe_sim_3d_ssf_note.exe ^
        -I"C:\vcpkg\installed\x64-windows\include" ^
        -L"C:\vcpkg\installed\x64-windows\lib" ^
        -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_ssf_note.cu -o dvripe_sim_3d_ssf_note \
 *       -lcufft -lGLEW -lglfw -lGL
 *
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

// For M_PI on MSVC
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Prevent GLFW from including gl.h
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

//---------------------------------------------------------------------------
// Simulation grid and PDE parameters
//---------------------------------------------------------------------------
static const int   NX   = 64;      // domain size in x
static const int   NY   = 64;      // domain size in y
static const int   NZ   = 64;      // domain size in z
static const float DX   = 0.1f;    // spatial step
static const float DT   = 0.0001f; // time step
static const float DVAL = 1.0f;    // dispersion coefficient
static const float GVAL = 1.0f;    // focusing strength
static const float NOTE = 1.0f;    // "Note" potential parameter
static const int   STEPS_PER_FRAME = 1; // PDE steps each rendered frame

//---------------------------------------------------------------------------
// GPU float2 helpers
//---------------------------------------------------------------------------
__device__ inline float2 cMul(const float2& a, const float2& b)
{
    // (a.x + i a.y)*(b.x + i b.y)
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ inline float cAbs2(const float2& a)
{
    return a.x*a.x + a.y*a.y;
}
__device__ inline float2 cExp(float phase)
{
    return make_float2(cosf(phase), sinf(phase));
}

//---------------------------------------------------------------------------
// "Note" Potential: A function that enforces a stable shape in real space.
//
// Example: We'll do a simple harmonic-like potential scaled by NOTE param:
//          V_note(r) = 0.5 * NOTE * r^2
// where r^2 = (x - cx)^2 + (y - cy)^2 + (z - cz)^2
// You can customize to any pattern-based shape or trap you like.
//---------------------------------------------------------------------------
__device__ float computeNotePotential(int i, int j, int k,
                                      int nx, int ny, int nz,
                                      float noteParam)
{
    // center
    float cx = nx/2.0f;
    float cy = ny/2.0f;
    float cz = nz/2.0f;

    float x = i - cx;
    float y = j - cy;
    float z = k - cz;

    float r2 = x*x + y*y + z*z;

    // A simple harmonic-like potential
    // V = 0.5 * noteParam * r^2
    // You can define any pattern or shape here.
    float V = 0.5f * noteParam * r2;
    return V;
}

//---------------------------------------------------------------------------
// We'll apply the Note potential in real space as an extra half-step:
//    ψ <- ψ * exp( -i V_note dt/2 )
//
// This kernel is analogous to the "nonlinear half-step" but for the potential.
//---------------------------------------------------------------------------
__global__ void applyNotePotentialHalfStepKernel(float2* psi,
                                                 int nx, int ny, int nz,
                                                 float dt, float noteParam)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if(i < nx && j < ny && k < nz)
    {
        int idx = (k*ny + j)*nx + i;
        float2 val = psi[idx];

        float V = computeNotePotential(i, j, k, nx, ny, nz, noteParam);

        // phase = - V dt/2
        float phase = - V * dt * 0.5f;

        float2 e = make_float2(cosf(phase), sinf(phase));
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;

        psi[idx] = out;
    }
}

//---------------------------------------------------------------------------
// PDE: i ∂ψ/∂t = -D ∇² ψ - G |ψ|² ψ + V_note(r)*ψ
// We'll do a Split-Step approach with potential included:
//    1) Nonlinear+Potential half-step
//    2) Linear step in Fourier space
//    3) Nonlinear+Potential half-step
//
// That means in real space we do exp( i [-G|ψ|^2 - V_note(r)] dt/2 ) each half-step.
// We separate the "nonlinear" and "notePotential" steps but combine them in one kernel
// or do them sequentially. For clarity, let's do them sequentially here.
//---------------------------------------------------------------------------
__global__ void nonlinearHalfStepKernel(float2* psi,
                                        int nx, int ny, int nz,
                                        float dt, float G)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if(i < nx && j < ny && k < nz)
    {
        int idx = (k*ny + j)*nx + i;
        float2 val = psi[idx];
        float amp2 = cAbs2(val);

        // focusing => i * [- G |psi|^2 ] => phase = - G amp2 dt/2
        float phase = - G*amp2 * (dt*0.5f);

        float2 e = make_float2(cosf(phase), sinf(phase));
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi[idx] = out;
    }
}

//---------------------------------------------------------------------------
// We'll do the linear step in Fourier space as usual
//---------------------------------------------------------------------------
__global__ void linearStepKernel(float2* psi_hat,
                                 int nx, int ny, int nz,
                                 float dt, float D,
                                 float dkx, float dky, float dkz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < nx*ny*nz)
    {
        int z = idx / (nx*ny);
        int r = idx % (nx*ny);
        int y = r / nx;
        int x = r % nx;

        // wave numbers
        int sx = (x < nx/2) ? x : (x - nx);
        int sy = (y < ny/2) ? y : (y - ny);
        int sz = (z < nz/2) ? z : (z - nz);

        float kx = sx*dkx;
        float ky = sy*dky;
        float kz = sz*dkz;
        float k2 = kx*kx + ky*ky + kz*kz;

        float phase = -D*k2*dt;
        float2 e = make_float2(cosf(phase), sinf(phase));

        float2 val = psi_hat[idx];
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi_hat[idx] = out;
    }
}

// Scale after inverse FFT
__global__ void scaleKernel(float2* data, float s, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n)
    {
        data[idx].x *= s;
        data[idx].y *= s;
    }
}

//---------------------------------------------------------------------------
// We'll compute amplitude in 3D for rendering the slice
//---------------------------------------------------------------------------
__global__ void computeAmplitude3D(float2* psi, float* amp,
                                   int nx, int ny, int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < nx*ny*nz)
    {
        float2 val = psi[idx];
        float amplitude = sqrtf(val.x*val.x + val.y*val.y);
        amp[idx] = amplitude;
    }
}

//---------------------------------------------------------------------------
// We'll fill a 2D slice z=NZ/2 for visualization
//---------------------------------------------------------------------------
__global__ void fillSliceKernel(const float2* psi,
                                unsigned char* slicePBO,
                                int nx, int ny, int nz, int sliceZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<nx && j<ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        float2 val = psi[idx3D];
        float amp = sqrtf(val.x*val.x + val.y*val.y);

        float scale = amp*10.0f;
        if(scale>1.0f) scale=1.0f;
        unsigned char c = (unsigned char)(scale*255.0f);

        int idx2D = j*nx + i;
        slicePBO[idx2D*4 + 0] = c;
        slicePBO[idx2D*4 + 1] = c;
        slicePBO[idx2D*4 + 2] = c;
        slicePBO[idx2D*4 + 3] = 255;
    }
}

//---------------------------------------------------------------------------
// We'll store the slice in a Pixel Buffer (PBO) for glDrawPixels
//---------------------------------------------------------------------------
static GLuint pboID;
static struct cudaGraphicsResource* pboResource = nullptr;

static bool createSlicePBO(int width, int height)
{
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&pboResource, pboID,
                               cudaGraphicsRegisterFlagsWriteDiscard);
    if(err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

static void renderSlicePBO(int width, int height)
{
    glRasterPos2i(0,0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

//---------------------------------------------------------------------------
// Minimal geometry for a single "slice" display in 2D
//---------------------------------------------------------------------------
static GLuint createWindow(int w, int h, const char* title)
{
    if(!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return 0;
    }
    GLFWwindow* window = glfwCreateWindow(w, h, title, nullptr, nullptr);
    if(!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 0;
    }
    glfwMakeContextCurrent(window);
    return 1;
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main()
{
    // Create a simple 2D window for slice display
    int winW=NX, winH=NY;
    if(!createWindow(winW, winH, "DVRIPE 3D SSF + Note Potential"))
    {
        return -1;
    }

    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK)
    {
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        return -1;
    }

    if(!createSlicePBO(NX, NY))
    {
        return -1;
    }

    // Allocate 3D field on host
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);

    // Initialize with a swirl
    float cx = NX/2.0f, cy=NY/2.0f, cz=NZ/2.0f;
    for(int z=0; z<NZ; z++)
    {
        for(int y=0; y<NY; y++)
        {
            for(int x=0; x<NX; x++)
            {
                int idx=(z*NY + y)*NX + x;
                float dx_ = x - cx;
                float dy_ = y - cy;
                float dz_ = z - cz;
                float r2  = dx_*dx_ + dy_*dy_ + dz_*dz_;
                float amp = expf(-r2/(NX*DX*0.3f));
                float th  = atan2f(dy_, dx_);
                psiHost[idx] = make_float2(amp*cosf(th), amp*sinf(th));
            }
        }
    }

    // Allocate on device
    float2* psiDev=nullptr;
    cudaMalloc(&psiDev, n*sizeof(float2));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(float2), cudaMemcpyHostToDevice);

    // PDE config
    cufftHandle plan;
    if(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C) != CUFFT_SUCCESS)
    {
        std::cerr << "cufftPlan3d failed!\n";
        return -1;
    }

    float dkx = 2.0f*M_PI/(NX*DX);
    float dky = 2.0f*M_PI/(NY*DX);
    float dkz = 2.0f*M_PI/(NZ*DX);

    // Kernel config
    dim3 threads(8,8,8);
    dim3 blocks((NX+threads.x-1)/threads.x,
                (NY+threads.y-1)/threads.y,
                (NZ+threads.z-1)/threads.z);

    int copyThreads=256;
    int copyBlocks=(n+copyThreads-1)/copyThreads;

    // For slice
    dim3 sliceThreads(16,16);
    dim3 sliceBlocks((NX+sliceThreads.x-1)/sliceThreads.x,
                     (NY+sliceThreads.y-1)/sliceThreads.y);

    // The main loop
    while(!glfwWindowShouldClose(glfwGetCurrentContext()))
    {
        // 1) Nonlinear + Note potential half-step
        //    i.e. apply nonlinearHalfStepKernel + applyNotePotentialHalfStepKernel
        //    in real space
        nonlinearHalfStepKernel<<<blocks, threads>>>(psiDev, NX, NY, NZ, DT, GVAL);
        cudaDeviceSynchronize();

        applyNotePotentialHalfStepKernel<<<blocks, threads>>>(psiDev,
                                                              NX, NY, NZ,
                                                              DT, NOTE);
        cudaDeviceSynchronize();

        // 2) Linear step in Fourier space
        cufftExecC2C(plan, (cufftComplex*)psiDev,
                             (cufftComplex*)psiDev, CUFFT_FORWARD);

        linearStepKernel<<<copyBlocks, copyThreads>>>(psiDev, NX, NY, NZ,
                                                      DT, DVAL,
                                                      dkx, dky, dkz);
        cudaDeviceSynchronize();

        cufftExecC2C(plan, (cufftComplex*)psiDev,
                             (cufftComplex*)psiDev, CUFFT_INVERSE);

        // scale
        float scaleVal=1.0f/(float)(NX*NY*NZ);
        scaleKernel<<<copyBlocks, copyThreads>>>(psiDev, scaleVal, n);
        cudaDeviceSynchronize();

        // 3) Nonlinear + Note potential half-step again
        nonlinearHalfStepKernel<<<blocks, threads>>>(psiDev, NX, NY, NZ, DT, GVAL);
        cudaDeviceSynchronize();

        applyNotePotentialHalfStepKernel<<<blocks, threads>>>(psiDev,
                                                              NX, NY, NZ,
                                                              DT, NOTE);
        cudaDeviceSynchronize();

        // 4) Fill slice z=NZ/2 into the PBO
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        unsigned char* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillSliceKernel<<<sliceBlocks, sliceThreads>>>(psiDev,
                                                       d_pbo,
                                                       NX, NY, NZ,
                                                       NZ/2);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &pboResource, 0);

        // 5) Render
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2i(0,0);
        renderSlicePBO(NX, NY);

        glfwSwapBuffers(glfwGetCurrentContext());
        glfwPollEvents();
    }

    // Cleanup
    cufftDestroy(plan);
    cudaFree(psiDev);

    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);

    glfwDestroyWindow(glfwGetCurrentContext());
    glfwTerminate();
    return 0;
}
