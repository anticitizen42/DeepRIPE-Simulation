/******************************************************************************
 * dvripe_sim_3d_ssf_realtime.cu
 *
 * Demonstrates a 3D Split-Step Fourier (SSF) integrator for the focusing
 * nonlinear Schrödinger equation:
 *
 *    i ∂ψ/∂t = -D ∇²ψ - G |ψ|² ψ
 *
 * with periodic boundaries, plus real-time rendering of a 2D slice (z=NZ/2)
 * in grayscale. This lets you visualize 3D DVRIPE vortex dynamics in real time
 * without the blow-up typical of naive Euler.
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_ssf_realtime.cu -o dvripe_sim_3d_ssf_realtime.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_ssf_realtime.cu -o dvripe_sim_3d_ssf_realtime \
 *       -lcufft -lGLEW -lglfw -lGL
 *
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>           // for cuFFT
#include <GL/glew.h>         // GLEW
#include <GLFW/glfw3.h>      // GLFW
#include <cuda_gl_interop.h> // CUDA-OpenGL interop

// For M_PI on MSVC
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------
// Simulation Parameters
// ---------------------
static const int NX = 64;
static const int NY = 64;
static const int NZ = 64;

// Spatial resolution
static const float DX = 0.1f;

// Time step
static const float DT = 0.001f;

// PDE coefficients
static const float DVAL = 1.0f; // Dispersion
static const float GVAL = 1.0f; // Focusing strength

// We'll do a certain number of PDE steps per frame
static const int STEPS_PER_FRAME = 1;

// We'll display the slice z=NZ/2, which is NX x NY
static const int SLICE_Z = NZ / 2;

// We'll do periodic boundaries in x,y,z

// ---------------------
// GPU Complex Type
// ---------------------
// We can use float2 for (real, imag). cufftComplex is also float2.
__device__ inline float2 cMul(const float2& a, const float2& b)
{
    return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ inline float2 cExp(float phase)
{
    return make_float2(cosf(phase), sinf(phase));
}
__device__ inline float cAbs2(const float2& a)
{
    return a.x*a.x + a.y*a.y;
}

// ---------------------------------------
// Kernel: Nonlinear Half-Step
// ψ <- ψ * exp(i * G * |ψ|^2 * dt/2)
// ---------------------------------------
__global__ void nonlinearHalfStepKernel(float2* psi, int n, float dt, float G)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float2 val = psi[idx];
        float amp2 = cAbs2(val);
        float phase = G * amp2 * (dt*0.5f);
        float2 e = make_float2(cosf(phase), sinf(phase));
        // multiply
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi[idx] = out;
    }
}

// ---------------------------------------
// Kernel: Multiply each Fourier mode
// by exp(-i * D * k^2 * dt)
// This is the linear step in Fourier space
// ---------------------------------------
__global__ void linearStepKernel(float2* psi_hat,
                                 int nx, int ny, int nz,
                                 float dt, float D,
                                 float dkx, float dky, float dkz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < (nx*ny*nz))
    {
        int z = idx / (nx*ny);
        int r = idx % (nx*ny);
        int y = r / nx;
        int x = r % nx;

        // wave numbers in each dimension
        // shift so that i in [-nx/2..nx/2-1], etc.
        int sx = (x < nx/2) ? x : (x - nx);
        int sy = (y < ny/2) ? y : (y - ny);
        int sz = (z < nz/2) ? z : (z - nz);

        float kx = sx*dkx;
        float ky = sy*dky;
        float kz = sz*dkz;
        float k2 = kx*kx + ky*ky + kz*kz;

        // multiply by exp(-i D k^2 dt)
        float phase = -D*k2*dt;
        float2 e = make_float2(cosf(phase), sinf(phase));

        float2 val = psi_hat[idx];
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi_hat[idx] = out;
    }
}

// ---------------------------------------
// Kernel: Scale the array by factor s
// (for inverse FFT normalization)
// ---------------------------------------
__global__ void scaleKernel(float2* data, float s, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n)
    {
        data[idx].x *= s;
        data[idx].y *= s;
    }
}

// ---------------------------------------
// We'll render amplitude of slice z=SLICE_Z
// into a Pixel Buffer (PBO) in grayscale
// ---------------------------------------
__global__ void fillSliceKernel(uchar4* pbo, float2* psi,
                                int nx, int ny, int nz, int sliceZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x; // x
    int j = blockIdx.y*blockDim.y + threadIdx.y; // y
    if(i < nx && j < ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        float2 val = psi[idx3D];
        float amp = sqrtf(val.x*val.x + val.y*val.y);

        // map amplitude -> grayscale
        float scale = amp*10.0f; // arbitrary
        if(scale>1.0f) scale=1.0f;
        unsigned char c = (unsigned char)(scale*255.0f);

        int idx2D = j*nx + i;
        pbo[idx2D].x = c;
        pbo[idx2D].y = c;
        pbo[idx2D].z = c;
        pbo[idx2D].w = 255;
    }
}

// ---------------------------------------
// Host code for initial condition
// We'll do a wide Gaussian * swirl
// in 3D
// ---------------------------------------
static void initializePsiHost(std::vector<float2>& psiHost,
                              int nx, int ny, int nz, float dx)
{
    float cx = nx/2.0f;
    float cy = ny/2.0f;
    float cz = nz/2.0f;

    for(int z=0; z<nz; z++){
        for(int y=0; y<ny; y++){
            for(int x=0; x<nx; x++){
                int idx = (z*ny + y)*nx + x;
                float dx_ = x - cx;
                float dy_ = y - cy;
                float dz_ = z - cz;
                float r2 = dx_*dx_ + dy_*dy_ + dz_*dz_;

                // amplitude => Gaussian
                float amp = expf(-r2/(nx*dx*0.3f));

                // swirl around z-axis? We'll do phase = atan2(y-ny/2, x-nx/2)
                float theta = atan2f(dy_, dx_);
                float re = amp*cosf(theta);
                float im = amp*sinf(theta);

                psiHost[idx] = make_float2(re, im);
            }
        }
    }
}

// ---------------------------------------
// CUDA-OpenGL interop
// We'll store a single 2D PBO for slice Nx x Ny
// ---------------------------------------
static GLuint pboID;
static struct cudaGraphicsResource* pboResource;

bool createPBO(int width, int height)
{
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&pboResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard);
    if(err != cudaSuccess){
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

// ---------------------------------------
// Main
// ---------------------------------------
int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    // We'll make a window Nx x Ny
    GLFWwindow* window = glfwCreateWindow(NX, NY, "3D DVRIPE SSF Real-Time", nullptr, nullptr);
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

    // 3) Allocate host memory for psi
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);
    initializePsiHost(psiHost, NX, NY, NZ, DX);

    // 4) Allocate device memory
    float2* psiDev;
    cudaMalloc(&psiDev, n*sizeof(float2));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(float2), cudaMemcpyHostToDevice);

    // 5) Create cuFFT 3D plan
    cufftHandle plan;
    // dimensions are (NZ, NY, NX)
    if(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C) != CUFFT_SUCCESS){
        std::cerr << "cufftPlan3d failed!\n";
        return -1;
    }

    // 6) Create PBO for slice
    if(!createPBO(NX, NY)){
        std::cerr << "Failed to create PBO\n";
        return -1;
    }

    // 7) define wave numbers increments
    // For a domain size = NX*DX in x, wave number step is (2π / (NX*DX)).
    float dkx = 2.0f*M_PI / (NX*DX);
    float dky = 2.0f*M_PI / (NY*DX);
    float dkz = 2.0f*M_PI / (NZ*DX);

    // 8) Kernel config
    int blockSize=256;
    int gridSize = (n + blockSize -1)/blockSize;

    dim3 threads(16,16);
    dim3 blocks( (NX+threads.x-1)/threads.x,
                 (NY+threads.y-1)/threads.y );

    // 9) Main loop
    while(!glfwWindowShouldClose(window))
    {
        // PDE steps
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // a) Nonlinear half-step
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();

            // b) Forward FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_FORWARD);

            // c) Linear step in Fourier space
            linearStepKernel<<<gridSize, blockSize>>>(psiDev, NX, NY, NZ, DT, DVAL, dkx, dky, dkz);
            cudaDeviceSynchronize();

            // d) Inverse FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_INVERSE);

            // e) scale by 1/(NX*NY*NZ)
            float scale = 1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<gridSize, blockSize>>>(psiDev, scale, n);
            cudaDeviceSynchronize();

            // f) Nonlinear half-step again
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();
        }

        // Copy slice to PBO
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        uchar4* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillSliceKernel<<<blocks, threads>>>(d_pbo, psiDev, NX, NY, NZ, SLICE_Z);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &pboResource, 0);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(0,0,NX,NY);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, NX, 0, NY, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        renderPBO(NX, NY);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cufftDestroy(plan);
    cudaFree(psiDev);

    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
