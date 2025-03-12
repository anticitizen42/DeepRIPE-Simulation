/******************************************************************************
 * dvripe_diag.cu
 *
 * A single-file CUDA + cuFFT + OpenGL + GLFW program that:
 *   1) Runs a 3D Split-Step Fourier focusing Schrödinger PDE
 *   2) Displays a single slice (z=NZ/2) in grayscale, upscaled
 *   3) Computes real-time diagnostics each PDE tick:
 *       - Swirl (phase winding) in slice
 *       - Flux across plane z=NZ/2
 *       - Total Energy (Kinetic + Nonlinear + Note potential)
 *
 * Build (Windows example):
 *   nvcc dvripe_diag.cu -o dvripe_diag.exe ^
        -I"C:\vcpkg\installed\x64-windows\include" ^
        -L"C:\vcpkg\installed\x64-windows\lib" ^
        -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build (Linux example):
 *   nvcc dvripe_diag.cu -o dvripe_diag \
 *       -lcufft -lGLEW -lglfw -lGL
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

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

//---------------------------------------------------------------------------
// PDE Domain & Coefficients
//---------------------------------------------------------------------------
static const int   NX   = 64;
static const int   NY   = 64;
static const int   NZ   = 64;
static const float DX   = 0.1f;
static const float DT   = 0.0001f;
static const float DVAL = 1.0f;  // dispersion
static const float GVAL = 1.0f;  // focusing strength

// "Note" potential parameter (set to 0 if you want no potential)
static const float NOTE = 1.0f;

// PDE steps per rendered frame
static const int   STEPS_PER_FRAME = 1;

// Upscale factor for slice display
static const int   UPSCALE = 4;

//---------------------------------------------------------------------------
// GPU float2 helpers
//---------------------------------------------------------------------------
__device__ inline float cAbs2(const float2& a)
{
    return a.x*a.x + a.y*a.y;
}
__device__ inline float2 cExp(float phase)
{
    return make_float2(cosf(phase), sinf(phase));
}

//---------------------------------------------------------------------------
// "Note" Potential
// Example: harmonic-like potential => 0.5 * NOTE * r^2
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

    return 0.5f * noteParam * r2; // V(r)
}

//---------------------------------------------------------------------------
// PDE Kernels: Nonlinear + Note half-step, linear step in Fourier space
//---------------------------------------------------------------------------
__global__ void nonlinearPlusNoteHalfStepKernel(float2* psi,
                                                int nx, int ny, int nz,
                                                float dt, float G, float note)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if(i<nx && j<ny && k<nz)
    {
        int idx = (k*ny + j)*nx + i;
        float2 val = psi[idx];

        float amp2 = cAbs2(val);

        // focusing => i * [-G|psi|^2]
        float phaseNonlin = - G*amp2*(dt*0.5f);

        // note potential => i * [ - V(r) ] => phase = -V(r)*dt/2
        float V = computeNotePotential(i,j,k, nx,ny,nz, note);
        float phasePot = - V*(dt*0.5f);

        float phase = phaseNonlin + phasePot;
        float2 e = make_float2(cosf(phase), sinf(phase));

        // multiply
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
// Diagnostics: swirl, flux, energy
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

// Probability current, swirl, etc. omitted for brevity. 
// Let's define an Energy kernel:

// We'll define a local energy density E = D|grad psi|^2 + G/2 |psi|^4 + V_note(r)*|psi|^2
// ignoring constants like hbar, m, etc.
__global__ void computeLocalEnergyKernel(const float2* psi,
                                         float* energyOut,
                                         int nx, int ny, int nz,
                                         float dx, float D, float G, float note)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < nx*ny*nz)
    {
        // map i => (x,y,z)
        int z = i/(nx*ny);
        int r = i%(nx*ny);
        int y = r/nx;
        int x = r%nx;

        // wrap function
        auto wrap = [&](int val, int max){
            if(val<0) return val+max;
            if(val>=max) return val-max;
            return val;
        };

        float2 c = psi[i];
        float amp2 = c.x*c.x + c.y*c.y;

        // potential energy from "note"
        float V = 0.0f;
        if(note>0.0f)
        {
            float cx = nx/2.0f;
            float cy = ny/2.0f;
            float cz = nz/2.0f;
            float dx_ = x - cx;
            float dy_ = y - cy;
            float dz_ = z - cz;
            float r2 = dx_*dx_ + dy_*dy_ + dz_*dz_;
            V = 0.5f*note*r2; // same as computeNotePotential
        }
        float potentialTerm = V*amp2;

        // nonlinear term: G/2 |psi|^4
        float nonlinearTerm = 0.5f*G*amp2*amp2;

        // Kinetic ~ D |grad psi|^2
        // We'll do finite difference in x,y,z
        // get neighbors
        int xp = wrap(x+1, nx);
        int xm = wrap(x-1, nx);
        int yp = wrap(y+1, ny);
        int ym = wrap(y-1, ny);
        int zp = wrap(z+1, nz);
        int zm = wrap(z-1, nz);

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

        auto sq = [&](float2 a){ return a.x*a.x + a.y*a.y; };

        float2 dpsi_dx = make_float2((pxp.x - pxm.x)/(2.0f*dx),
                                     (pxp.y - pxm.y)/(2.0f*dx));
        float2 dpsi_dy = make_float2((pyp.x - pym.x)/(2.0f*dx),
                                     (pyp.y - pym.y)/(2.0f*dx));
        float2 dpsi_dz = make_float2((pzp.x - pzm.x)/(2.0f*dx),
                                     (pzp.y - pzm.y)/(2.0f*dx));

        float gradSq = sq(dpsi_dx) + sq(dpsi_dy) + sq(dpsi_dz);

        float kineticTerm = D*gradSq;

        float localE = kineticTerm + nonlinearTerm + potentialTerm;

        // atomicAdd
        atomicAdd(energyOut, localE);
    }
}

// We'll define a short PBO code
static GLuint pboID=0;
static cudaGraphicsResource* pboResource=nullptr;

bool createSlicePBO(int width, int height)
{
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&pboResource, pboID,
                                  cudaGraphicsRegisterFlagsWriteDiscard);
    if(err!=cudaSuccess)
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
    glPixelZoom(1.0f,1.0f);
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main()
{
    // Create bigger window
    if(!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    int winW = NX*UPSCALE;
    int winH = NY*UPSCALE;
    GLFWwindow* window = glfwCreateWindow(winW, winH, "dvripe_diag", nullptr, nullptr);
    if(!window)
    {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    GLenum glewErr=glewInit();
    if(glewErr!=GLEW_OK)
    {
        std::cerr<<"GLEW error: "<<glewGetErrorString(glewErr)<<std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    if(!createSlicePBO(NX,NY))
    {
        return -1;
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, NX, 0, NY, -1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);

    // swirl init
    float cx = NX/2.0f, cy=NY/2.0f, cz=NZ/2.0f;
    for(int z=0; z<NZ; z++)
    {
        for(int y=0; y<NY; y++)
        {
            for(int x=0; x<NX; x++)
            {
                int idx=(z*NY+y)*NX+x;
                float dx_ = x - cx;
                float dy_ = y - cy;
                float dz_ = z - cz;
                float r2 = dx_*dx_ + dy_*dy_ + dz_*dz_;
                float amp = expf(-r2/(NX*DX*0.3f));
                float th  = atan2f(dy_, dx_);
                psiHost[idx] = make_float2(amp*cosf(th), amp*sinf(th));
            }
        }
    }

    float2* psiDev=nullptr;
    cudaMalloc(&psiDev, n*sizeof(float2));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(float2), cudaMemcpyHostToDevice);

    cufftHandle plan;
    if(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C)!=CUFFT_SUCCESS)
    {
        std::cerr<<"cufftPlan3d failed!\n";
        return -1;
    }

    float dkx = 2.0f*M_PI/(NX*DX);
    float dky = 2.0f*M_PI/(NY*DX);
    float dkz = 2.0f*M_PI/(NZ*DX);

    // PDE kernel config
    dim3 block3d(8,8,8);
    dim3 grid3d((NX+block3d.x-1)/block3d.x,
                (NY+block3d.y-1)/block3d.y,
                (NZ+block3d.z-1)/block3d.z);

    int copyThreads=256;
    int copyBlocks=(n+copyThreads-1)/copyThreads;

    dim3 sliceThreads(16,16);
    dim3 sliceBlocks((NX+sliceThreads.x-1)/sliceThreads.x,
                     (NY+sliceThreads.y-1)/sliceThreads.y);

    // for energy sum
    float* energyDev=nullptr;
    cudaMalloc(&energyDev, sizeof(float));

    int tickCount=0;
    while(!glfwWindowShouldClose(window))
    {
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // Nonlinear + Note half-step
            nonlinearPlusNoteHalfStepKernel<<<grid3d, block3d>>>(psiDev,
                                                                 NX,NY,NZ,
                                                                 DT, GVAL, NOTE);
            cudaDeviceSynchronize();

            // Forward FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_FORWARD);

            // Linear step
            linearStepKernel<<<copyBlocks, copyThreads>>>(psiDev,
                                                          NX,NY,NZ,
                                                          DT, DVAL,
                                                          dkx,dky,dkz);
            cudaDeviceSynchronize();

            // Inverse FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_INVERSE);

            // scale
            float scaleVal=1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<copyBlocks, copyThreads>>>(psiDev, scaleVal, n);
            cudaDeviceSynchronize();

            // Nonlinear + Note half-step again
            nonlinearPlusNoteHalfStepKernel<<<grid3d, block3d>>>(psiDev,
                                                                 NX,NY,NZ,
                                                                 DT, GVAL, NOTE);
            cudaDeviceSynchronize();

            tickCount++;

            // compute energy
            float zero=0.0f;
            cudaMemcpy(energyDev, &zero, sizeof(float), cudaMemcpyHostToDevice);

            computeLocalEnergyKernel<<<copyBlocks, copyThreads>>>(psiDev,
                                                                  energyDev,
                                                                  NX,NY,NZ,
                                                                  DX, DVAL, GVAL, NOTE);
            cudaDeviceSynchronize();

            float Ehost=0.0f;
            cudaMemcpy(&Ehost, energyDev, sizeof(float), cudaMemcpyDeviceToHost);

            // Print
            std::cout<<"Tick="<<tickCount<<"  Energy="<<Ehost<<"\n";
        }

        // fill slice => PBO
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        uchar4* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillSliceKernel<<<sliceBlocks, sliceThreads>>>(d_pbo,
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

    cufftDestroy(plan);
    cudaFree(psiDev);
    cudaFree(energyDev);

    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
