/******************************************************************************
 * dvripe_sim_3d_ssf_note_hires.cu
 *
 * A single-file CUDA + cuFFT + OpenGL + GLFW program that:
 *   1) Runs a 3D Split-Step Fourier focusing Schrödinger PDE with a "Note"
 *      potential (trap/pattern).
 *   2) Uses a higher resolution domain (128x128x64) for more detail.
 *   3) Displays the amplitude slice (z=NZ/2) in a larger window with
 *      pixel zoom for a bigger, more defined output.
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_ssf_note_hires.cu -o dvripe_sim_3d_ssf_note_hires.exe ^
        -I"C:\vcpkg\installed\x64-windows\include" ^
        -L"C:\vcpkg\installed\x64-windows\lib" ^
        -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_ssf_note_hires.cu -o dvripe_sim_3d_ssf_note_hires \
 *       -lcufft -lGLEW -lglfw -lGL
 *
 * Notes:
 *   - NX=128, NY=128, NZ=64 => 1,048,576 cells, heavier on GPU.
 *   - The window is set to (NX*UPSCALE) x (NY*UPSCALE), plus glPixelZoom.
 *   - If performance is slow or GPU memory is insufficient, reduce domain size.
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

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
// Domain & PDE parameters
//---------------------------------------------------------------------------
static const int   NX    = 128;     // bigger resolution in x
static const int   NY    = 128;     // bigger resolution in y
static const int   NZ    = 64;      // somewhat large in z

static const float DX    = 0.1f;    // spatial step
static const float DT    = 0.0001f; // time step
static const float DVAL  = 0.5f;    // dispersion
static const float GVAL  = 2.0f;    // focusing strength
static const float NOTE  = 1.0f;    // "Note" potential parameter
static const int   STEPS_PER_FRAME = 1;

// We’ll upscale the slice display by 6x
static const int   UPSCALE = 6;

//---------------------------------------------------------------------------
// GPU float2 helpers
//---------------------------------------------------------------------------
__device__ inline float2 cMul(const float2& a, const float2& b)
{
    // (a.x + i a.y)*(b.x + i b.y)
    return make_float2(a.x*b.x - a.y*b.y,
                       a.x*b.y + a.y*b.x);
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
// "Note" Potential: harmonic-like for demonstration
//---------------------------------------------------------------------------
__device__ float computeNotePotential(int i, int j, int k,
                                      int nx, int ny, int nz,
                                      float noteParam)
{
    float cx = nx/2.0f;
    float cy = ny/2.0f;
    float cz = nz/2.0f;

    float x = i - cx;
    float y = j - cy;
    float z = k - cz;
    float r2 = x*x + y*y + z*z;

    // V = 0.5 * noteParam * r^2
    float V = 0.5f * noteParam * r2;
    return V;
}

//---------------------------------------------------------------------------
// We'll apply the Note potential in real space as an extra half-step
//    ψ <- ψ * exp(-i V_note dt/2)
//---------------------------------------------------------------------------
__global__ void applyNotePotentialHalfStep(float2* psi,
                                           int nx, int ny, int nz,
                                           float dt, float noteParam)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if(i<nx && j<ny && k<nz)
    {
        int idx = (k*ny + j)*nx + i;
        float2 val = psi[idx];

        float V = computeNotePotential(i,j,k, nx,ny,nz, noteParam);
        float phase = - V * dt * 0.5f;

        float2 e = make_float2(cosf(phase), sinf(phase));
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi[idx] = out;
    }
}

//---------------------------------------------------------------------------
// Nonlinear half-step: focusing
//    ψ <- ψ * exp(i * [-G|ψ|^2] dt/2)
//---------------------------------------------------------------------------
__global__ void nonlinearHalfStep(float2* psi,
                                  int nx, int ny, int nz,
                                  float dt, float G)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if(i<nx && j<ny && k<nz)
    {
        int idx = (k*ny + j)*nx + i;
        float2 val = psi[idx];
        float amp2 = cAbs2(val);

        // focusing => phase = - G amp2 dt/2
        float phase = - G*amp2 * (dt*0.5f);

        float2 e = make_float2(cosf(phase), sinf(phase));
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi[idx] = out;
    }
}

//---------------------------------------------------------------------------
// Linear step in Fourier space
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

// scale after inverse FFT
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
// We'll fill a PBO with the amplitude slice z=NZ/2
// Each pixel is RGBA (uchar4). We'll do a grayscale mapping of amplitude.
//---------------------------------------------------------------------------
__global__ void fillSliceKernel(uchar4* pbo,
                                const float2* psi,
                                int nx, int ny, int nz, int sliceZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<nx && j<ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        float2 val = psi[idx3D];
        float amp = sqrtf(val.x*val.x + val.y*val.y);

        // arbitrary scaling
        float scale = amp*10.0f;
        if(scale>1.0f) scale=1.0f;
        unsigned char c = (unsigned char)(scale*255.0f);

        int idx2D = j*nx + i;
        pbo[idx2D] = make_uchar4(c, c, c, 255);
    }
}

//---------------------------------------------------------------------------
// We'll store the slice in a Pixel Buffer (PBO)
//---------------------------------------------------------------------------
static GLuint pboID=0;
static struct cudaGraphicsResource* pboResource=nullptr;

bool createSlicePBO(int width, int height)
{
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&pboResource,
                                                   pboID,
                                                   cudaGraphicsRegisterFlagsWriteDiscard);
    if(err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

void renderSlicePBO(int width, int height)
{
    // We'll do a glDrawPixels with pixel zoom
    glPixelZoom((float)UPSCALE, (float)UPSCALE);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // reset pixel zoom
    glPixelZoom(1.0f, 1.0f);
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main()
{
    // 1) Create a bigger window
    if(!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    int winW = NX*UPSCALE;
    int winH = NY*UPSCALE;
    GLFWwindow* window = glfwCreateWindow(winW, winH,
                                          "DVRIPE 3D SSF + Note Potential - HiRes",
                                          nullptr, nullptr);
    if(!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 2) Init GLEW
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK)
    {
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 3) Create PBO for slice
    if(!createSlicePBO(NX, NY))
    {
        std::cerr << "Failed to create slice PBO\n";
        return -1;
    }

    // 4) Orthographic 2D for drawing the slice
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // from (0..NX, 0..NY)
    glOrtho(0, NX, 0, NY, -1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 5) Allocate field on host
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);

    // swirl initial
    float cx = NX/2.0f;
    float cy = NY/2.0f;
    float cz = NZ/2.0f;
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

    // 6) Allocate on device
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

    dim3 threads(8,8,8);
    dim3 blocks((NX+threads.x-1)/threads.x,
                (NY+threads.y-1)/threads.y,
                (NZ+threads.z-1)/threads.z);

    int copyThreads=256;
    int copyBlocks=(n+copyThreads-1)/copyThreads;

    dim3 sliceThreads(16,16);
    dim3 sliceBlocks((NX+sliceThreads.x-1)/sliceThreads.x,
                     (NY+sliceThreads.y-1)/sliceThreads.y);

    // Device function to combine "nonlinear" + "note potential" in real space
    // We'll do them sequentially for clarity
    auto doNonlinearNoteHalfStep = [&](float2* psiDev)
    {
        // 1) Nonlinear half-step
        nonlinearHalfStep<<<blocks, threads>>>(psiDev, NX, NY, NZ, DT, GVAL);
        cudaDeviceSynchronize();

        // 2) Note potential half-step
        applyNotePotentialHalfStep<<<blocks, threads>>>(psiDev,
                                                        NX, NY, NZ,
                                                        DT, NOTE);
        cudaDeviceSynchronize();
    };

    while(!glfwWindowShouldClose(window))
    {
        // PDE steps
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // Nonlinear + note half-step
            doNonlinearNoteHalfStep(psiDev);

            // Linear step in Fourier space
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_FORWARD);

            linearStepKernel<<<copyBlocks, copyThreads>>>(psiDev,
                                                          NX, NY, NZ,
                                                          DT, DVAL,
                                                          dkx, dky, dkz);
            cudaDeviceSynchronize();

            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_INVERSE);

            // scale
            float scaleVal = 1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<copyBlocks, copyThreads>>>(psiDev, scaleVal, n);
            cudaDeviceSynchronize();

            // Nonlinear + note half-step again
            doNonlinearNoteHalfStep(psiDev);
        }

        // fill slice z=NZ/2
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        uchar4* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillSliceKernel<<<sliceBlocks, sliceThreads>>>(d_pbo,
                                                       psiDev,
                                                       NX, NY, NZ,
                                                       NZ/2);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &pboResource, 0);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glRasterPos2f(0,0);
        renderSlicePBO(NX, NY);

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
