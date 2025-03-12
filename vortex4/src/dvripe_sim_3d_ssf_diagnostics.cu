/******************************************************************************
 * dvripe_sim_3d_ssf_diagnostics.cu
 *
 * Demonstrates a 3D Split-Step Fourier focusing Schrödinger PDE in CUDA,
 * plus real-time diagnostics:
 *   1) Probability Current (mass-energy flow)
 *   2) Net Flux across a plane
 *   3) Phase Winding in a 2D slice
 *   4) Chirality (sign of swirl)
 *
 * Each PDE time step is considered a "tick" of the Compton Clock. After each
 * step, we compute and print these diagnostics. We also display a single
 * slice (z=NZ/2) in grayscale.
 *
 * Corrected version: 
 *   - `computeSliceSwirlKernel` has 4 arguments
 *   - `fillSliceKernel` is defined
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
static const int   NX   = 64;
static const int   NY   = 64;
static const int   NZ   = 64;
static const float DX   = 0.1f;
static const float DT   = 0.0005f;
static const float DVAL = 1.0f;
static const float GVAL = 1.0f; // focusing
static const int   STEPS_PER_FRAME = 1;

static const int   UPSCALE = 4; // upscaling the slice display

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
// PDE kernels (focusing PDE, no note potential for brevity)
//---------------------------------------------------------------------------
__global__ void nonlinearHalfStepKernel(float2* psi, int n, float dt, float G)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float2 val = psi[idx];
        float amp2 = cAbs2(val);
        // focusing => i * [-G|psi|^2], half-step => phase = -G amp2 dt/2
        float phase = - G*amp2*(dt*0.5f);
        float2 e = make_float2(cosf(phase), sinf(phase));
        float2 out;
        out.x = val.x*e.x - val.y*e.y;
        out.y = val.x*e.y + val.y*e.x;
        psi[idx] = out;
    }
}

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
// Probability Current: J = Im(psi^* grad(psi))
//---------------------------------------------------------------------------
__global__ void computeProbabilityCurrentKernel(const float2* psi,
                                                float3* J, // Jx,Jy,Jz
                                                int nx, int ny, int nz,
                                                float dx)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < nx*ny*nz)
    {
        // index => (x,y,z)
        int z = i/(nx*ny);
        int r = i%(nx*ny);
        int y = r/nx;
        int x = r%nx;

        auto wrap = [&](int val, int max){
            if(val<0)    return val+max;
            if(val>=max) return val-max;
            return val;
        };

        float2 c = psi[i];

        int xp = wrap(x+1,nx), xm=wrap(x-1,nx);
        int yp = wrap(y+1,ny), ym=wrap(y-1,ny);
        int zp = wrap(z+1,nz), zm=wrap(z-1,nz);

        int idx_xp = (z*ny + y)*nx + xp;
        int idx_xm = (z*ny + y)*nx + xm;
        int idx_yp = (z*ny + yp)*nx + x;
        int idx_ym = (z*ny + ym)*nx + x;
        int idx_zp = ((zp)*ny + y)*nx + x;
        int idx_zm = ((zm)*ny + y)*nx + x;

        float2 pxp = psi[idx_xp];
        float2 pxm = psi[idx_xm];
        float2 pyp = psi[idx_yp];
        float2 pym = psi[idx_ym];
        float2 pzp = psi[idx_zp];
        float2 pzm = psi[idx_zm];

        // grad(psi)
        float2 dpsi_dx = make_float2((pxp.x - pxm.x)/(2.0f*dx),
                                     (pxp.y - pxm.y)/(2.0f*dx));
        float2 dpsi_dy = make_float2((pyp.x - pym.x)/(2.0f*dx),
                                     (pyp.y - pym.y)/(2.0f*dx));
        float2 dpsi_dz = make_float2((pzp.x - pzm.x)/(2.0f*dx),
                                     (pzp.y - pzm.y)/(2.0f*dx));

        float2 cstar = make_float2(c.x, -c.y);

        auto ImMul = [&](float2 a, float2 b){
            // Imag part of (a*b)
            // a*b = (a.x*b.x - a.y*b.y) + i(a.x*b.y + a.y*b.x)
            return a.x*b.y + a.y*b.x;
        };

        float Jx = ImMul(cstar, dpsi_dx);
        float Jy = ImMul(cstar, dpsi_dy);
        float Jz = ImMul(cstar, dpsi_dz);

        J[i] = make_float3(Jx,Jy,Jz);
    }
}

// net flux across plane z=planeZ => sum of Jz
__global__ void sumFluxKernel(const float3* J,
                              float* fluxOut,
                              int nx, int ny, int nz,
                              int planeZ)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x<nx && y<ny)
    {
        int idx3D = (planeZ*ny + y)*nx + x;
        float val = J[idx3D].z;
        atomicAdd(fluxOut, val);
    }
}

//---------------------------------------------------------------------------
// Slice swirl & chirality
// We'll do two steps:
//   1) computeSlicePhaseKernel => fill phaseSlice
//   2) computeSliceSwirlKernel => sum swirl
//---------------------------------------------------------------------------
__global__ void computeSlicePhaseKernel(const float2* psi,
                                        float* phaseSlice,
                                        int nx, int ny, int nz,
                                        int sliceZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<nx && j<ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        float2 val = psi[idx3D];
        float ph = atan2f(val.y, val.x);
        int idx2D = j*nx + i;
        phaseSlice[idx2D] = ph;
    }
}

// We'll do a net swirl approach: sum swirl in each 2x2 cell
__global__ void computeSliceSwirlKernel(const float* phaseSlice,
                                        float* swirlOut,
                                        int nx, int ny)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i < nx-1 && j < ny-1)
    {
        int idx00 = j*nx + i;
        int idx10 = j*nx + (i+1);
        int idx01 = (j+1)*nx + i;
        int idx11 = (j+1)*nx + (i+1);

        float p00 = phaseSlice[idx00];
        float p10 = phaseSlice[idx10];
        float p01 = phaseSlice[idx01];
        float p11 = phaseSlice[idx11];

        auto unwrapDiff = [&](float a, float b){
            float d = b - a;
            while(d> M_PI) d -= 2.0f*M_PI;
            while(d<-M_PI) d += 2.0f*M_PI;
            return d;
        };

        float d1 = unwrapDiff(p00, p10);
        float d2 = unwrapDiff(p10, p11);
        float d3 = unwrapDiff(p11, p01);
        float d4 = unwrapDiff(p01, p00);
        float swirl = d1 + d2 + d3 + d4;

        atomicAdd(swirlOut, swirl);
    }
}

//---------------------------------------------------------------------------
// We'll define a kernel to fill the slice into a PBO
//---------------------------------------------------------------------------
__global__ void fillSliceKernel(uchar4* pbo,
                                const float2* psi,
                                int nx, int ny, int nz,
                                int sliceZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<nx && j<ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        float2 val = psi[idx3D];
        float amp = sqrtf(val.x*val.x + val.y*val.y);

        float scale = amp*10.0f; // arbitrary
        if(scale>1.0f) scale=1.0f;
        unsigned char c = (unsigned char)(scale*255.0f);

        int idx2D = j*nx + i;
        pbo[idx2D] = make_uchar4(c,c,c,255);
    }
}

//---------------------------------------------------------------------------
// We'll define a PBO for the slice
//---------------------------------------------------------------------------
static GLuint pboID=0;
static cudaGraphicsResource* pboResource=nullptr;

bool createSlicePBO(int width, int height)
{
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*sizeof(uchar4),
                 nullptr, GL_DYNAMIC_DRAW);
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
    glPixelZoom((float)UPSCALE, (float)UPSCALE);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glPixelZoom(1.0f, 1.0f);
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main()
{
    // 1) Create bigger window
    if(!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    int winW = NX*UPSCALE;
    int winH = NY*UPSCALE;
    GLFWwindow* window = glfwCreateWindow(winW, winH,
                                          "DVRIPE 3D SSF + Diagnostics (Fixed)",
                                          nullptr, nullptr);
    if(!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 2) GLEW
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

    // orthographic 2D
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, NX, 0, NY, -1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // 4) Allocate field on host
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
                int idx = (z*NY + y)*NX + x;
                float dx_ = x - cx;
                float dy_ = y - cy;
                float dz_ = z - cz;
                float r2  = dx_*dx_ + dy_*dy_ + dz_*dz_;
                float amp = expf(-r2/(NX*0.1f*0.3f));
                float th  = atan2f(dy_, dx_);
                psiHost[idx] = make_float2(amp*cosf(th), amp*sinf(th));
            }
        }
    }

    // 5) Allocate on device
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

    float dkx = 2.0f*M_PI/(NX*0.1f);
    float dky = 2.0f*M_PI/(NY*0.1f);
    float dkz = 2.0f*M_PI/(NZ*0.1f);

    int copyThreads=256;
    int copyBlocks=(n+copyThreads-1)/copyThreads;

    dim3 sliceThreads(16,16);
    dim3 sliceBlocks((NX+sliceThreads.x-1)/sliceThreads.x,
                     (NY+sliceThreads.y-1)/sliceThreads.y);

    // Probability current arrays
    float3* Jdev=nullptr;
    cudaMalloc(&Jdev, n*sizeof(float3));

    // single float for flux, swirl
    float* fluxDev=nullptr;
    cudaMalloc(&fluxDev, sizeof(float));

    float* swirlDev=nullptr;
    cudaMalloc(&swirlDev, sizeof(float));

    // slice phase array
    float* phaseSliceDev=nullptr;
    cudaMalloc(&phaseSliceDev, NX*NY*sizeof(float));

    // PDE loop
    int tickCount=0;
    while(!glfwWindowShouldClose(window))
    {
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // Nonlinear half-step
            nonlinearHalfStepKernel<<<copyBlocks, copyThreads>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();

            // Forward FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_FORWARD);

            // Linear step
            linearStepKernel<<<copyBlocks, copyThreads>>>(psiDev,
                                                          NX, NY, NZ,
                                                          DT, DVAL,
                                                          dkx, dky, dkz);
            cudaDeviceSynchronize();

            // Inverse
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_INVERSE);

            // scale
            float scaleVal = 1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<copyBlocks, copyThreads>>>(psiDev, scaleVal, n);
            cudaDeviceSynchronize();

            // Nonlinear half-step again
            nonlinearHalfStepKernel<<<copyBlocks, copyThreads>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();

            tickCount++;

            // A) Probability current J
            {
                dim3 block3d(8,8,8);
                dim3 grid3d((NX+block3d.x-1)/block3d.x,
                            (NY+block3d.y-1)/block3d.y,
                            (NZ+block3d.z-1)/block3d.z);
                computeProbabilityCurrentKernel<<<grid3d, block3d>>>(
                    psiDev, Jdev, NX,NY,NZ, 0.1f);
                cudaDeviceSynchronize();
            }

            // B) Net flux across plane z=NZ/2 => sum J.z
            {
                float zero=0.0f;
                cudaMemcpy(fluxDev, &zero, sizeof(float), cudaMemcpyHostToDevice);

                dim3 block2d(16,16);
                dim3 grid2d((NX+block2d.x-1)/block2d.x,
                            (NY+block2d.y-1)/block2d.y);
                sumFluxKernel<<<grid2d, block2d>>>(Jdev,
                                                   fluxDev,
                                                   NX,NY,NZ,
                                                   NZ/2);
                cudaDeviceSynchronize();

                float fluxHost=0.0f;
                cudaMemcpy(&fluxHost, fluxDev, sizeof(float), cudaMemcpyDeviceToHost);

                // print later
                std::cout << "Tick=" << tickCount
                          << "  fluxZ=" << fluxHost;
            }

            // C) Slice swirl & chirality
            {
                // compute slice phase
                dim3 block2d(16,16);
                dim3 grid2d((NX+block2d.x-1)/block2d.x,
                            (NY+block2d.y-1)/block2d.y);

                computeSlicePhaseKernel<<<grid2d, block2d>>>(
                    psiDev, phaseSliceDev, NX,NY,NZ, NZ/2);
                cudaDeviceSynchronize();

                float zero=0.0f;
                cudaMemcpy(swirlDev, &zero, sizeof(float), cudaMemcpyHostToDevice);

                computeSliceSwirlKernel<<<grid2d, block2d>>>(
                    phaseSliceDev, swirlDev, NX,NY);
                cudaDeviceSynchronize();

                float swirlHost=0.0f;
                cudaMemcpy(&swirlHost, swirlDev, sizeof(float), cudaMemcpyDeviceToHost);

                int chirality=0;
                if(swirlHost>0.0f) chirality=+1;
                if(swirlHost<0.0f) chirality=-1;

                std::cout << "  swirl=" << swirlHost
                          << "  chirality=" << chirality
                          << std::endl;
            }
        }

        // fill slice => PBO
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        uchar4* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        dim3 block2d(16,16);
        dim3 grid2d((NX+block2d.x-1)/block2d.x,
                    (NY+block2d.y-1)/block2d.y);
        fillSliceKernel<<<grid2d, block2d>>>(d_pbo,
                                             psiDev,
                                             NX,NY,NZ,
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
    cudaFree(Jdev);
    cudaFree(fluxDev);
    cudaFree(swirlDev);
    cudaFree(phaseSliceDev);

    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
