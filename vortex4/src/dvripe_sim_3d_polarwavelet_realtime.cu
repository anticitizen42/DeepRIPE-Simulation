/******************************************************************************
 * dvripe_sim_3d_polarwavelet_realtime.cu
 *
 * Demonstrates a 3D focusing PDE that collapses into a "dot," plus a
 * dual-radius polar discretization + wavelet transform in angle.
 *
 * We show three panels in one window:
 *   1) Left : The 2D slice amplitude (z = NZ/2).
 *   2) Center: Wavelet scalogram (scale vs. angle) for radius R1.
 *   3) Right : Wavelet scalogram (scale vs. angle) for radius R2.
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_polarwavelet_realtime.cu -o dvripe_sim_3d_polarwavelet_realtime.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lglew32 -lglfw3dll -lopengl32
 *
 *   (Adjust library names as needed, e.g. -lglfw3 for static link.)
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_polarwavelet_realtime.cu -o dvripe_sim_3d_polarwavelet_realtime \
 *       -lGLEW -lglfw -lGL
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Ensure M_PI is available
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
// 3D PDE Parameters
// -----------------------------------------------------------------------------
static const int NX = 64;  // grid size in x
static const int NY = 64;  // grid size in y
static const int NZ = 64;  // grid size in z

static const float DX = 0.1f;    // spatial step
static const float DT = 0.0001f; // time step
static const float D  = 1.0f;    // diffusion/dispersion coefficient
static const float G  = 1.0f;    // focusing strength

static const int STEPS_PER_FRAME = 1; // PDE steps per rendered frame

// We'll show the slice z=NZ/2 in the left panel
static const int SLICE_Z = NZ/2;

// -----------------------------------------------------------------------------
// "Dual" radii for polar wavelet analysis
// -----------------------------------------------------------------------------
static const float R1 = 3.0f;   // inner radius
static const float R2 = 6.0f;   // outer radius
static const int   NUM_ANGLES = 512; // fine angular resolution
static const int   NUM_SCALES = 64;  // wavelet scales
static const float W0 = 6.0f;       // Morlet wavelet carrier freq

// -----------------------------------------------------------------------------
// 3D complex field
// -----------------------------------------------------------------------------
struct Complex3D {
    float x; // real
    float y; // imag
};

__device__ inline float cAbs2(const Complex3D& c)
{
    return c.x*c.x + c.y*c.y;
}

// -----------------------------------------------------------------------------
// PDE: ∂ψ/∂t = i [ D ∇²ψ - G |ψ|^2 ψ ]
// Focusing because of the -G term
// -----------------------------------------------------------------------------
__global__ void pdeUpdateKernel3D(Complex3D* psi, Complex3D* psiNew,
                                  int nx, int ny, int nz,
                                  float dx, float dt,
                                  float Dval, float Gval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y
    int k = blockIdx.z * blockDim.z + threadIdx.z; // z

    if(i < nx && j < ny && k < nz)
    {
        // Flatten 3D -> 1D
        int idx = (k * ny + j)*nx + i;

        auto wrap = [&](int c, int maxVal){
            if(c < 0) c += maxVal;
            if(c >= maxVal) c -= maxVal;
            return c;
        };
        int ip = wrap(i+1, nx);
        int im = wrap(i-1, nx);
        int jp = wrap(j+1, ny);
        int jm = wrap(j-1, ny);
        int kp = wrap(k+1, nz);
        int km = wrap(k-1, nz);

        int idx_ip = (k*ny + j)*nx + ip;
        int idx_im = (k*ny + j)*nx + im;
        int idx_jp = (k*ny + jp)*nx + i;
        int idx_jm = (k*ny + jm)*nx + i;
        int idx_kp = (kp*ny + j)*nx + i;
        int idx_km = (km*ny + j)*nx + i;

        // Laplacian in 3D
        Complex3D lap;
        lap.x = ( psi[idx_ip].x + psi[idx_im].x
                + psi[idx_jp].x + psi[idx_jm].x
                + psi[idx_kp].x + psi[idx_km].x
                - 6.0f*psi[idx].x ) / (dx*dx);
        lap.y = ( psi[idx_ip].y + psi[idx_im].y
                + psi[idx_jp].y + psi[idx_jm].y
                + psi[idx_kp].y + psi[idx_km].y
                - 6.0f*psi[idx].y ) / (dx*dx);

        // i * D * lap
        Complex3D linear;
        linear.x = -lap.y * Dval; // multiply by i => -y
        linear.y =  lap.x * Dval; //                 x

        // i * [-G|ψ|^2] * ψ
        float amp2 = cAbs2(psi[idx]);
        Complex3D nonlin;
        nonlin.x = -psi[idx].y; // multiply by i
        nonlin.y =  psi[idx].x;
        float factor = -Gval * amp2;  // minus sign for focusing
        nonlin.x *= factor;
        nonlin.y *= factor;

        // dpsi/dt
        Complex3D dpsi_dt;
        dpsi_dt.x = linear.x + nonlin.x;
        dpsi_dt.y = linear.y + nonlin.y;

        // Euler step
        psiNew[idx].x = psi[idx].x + dt*dpsi_dt.x;
        psiNew[idx].y = psi[idx].y + dt*dpsi_dt.y;
    }
}

// Copy kernel
__global__ void copyKernel3D(Complex3D* psi, Complex3D* psiNew, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        psi[idx] = psiNew[idx];
    }
}

// -----------------------------------------------------------------------------
// Initialize field: wide Gaussian
// -----------------------------------------------------------------------------
void initializeField3D(std::vector<Complex3D>& psiHost, int nx, int ny, int nz, float dx)
{
    float cx = nx/2.0f;
    float cy = ny/2.0f;
    float cz = nz/2.0f;

    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                int idx = (k*ny + j)*nx + i;
                float x = i - cx;
                float y = j - cy;
                float z = k - cz;
                float r2 = x*x + y*y + z*z;

                // wide Gaussian
                float amplitude = expf(-r2 / (nx*dx*0.3f));
                psiHost[idx].x = amplitude; // purely real
                psiHost[idx].y = 0.0f;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 2D slice kernel: fill a PBO with amplitude from z=SLICE_Z
// -----------------------------------------------------------------------------
__global__ void fillSliceKernel(uchar4* pbo,
                                const Complex3D* psi,
                                int nx, int ny, int nz, int sliceZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y

    if(i < nx && j < ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        int idx2D = j*nx + i;

        float amp = sqrtf(psi[idx3D].x*psi[idx3D].x + psi[idx3D].y*psi[idx3D].y);

        // grayscale
        float val = amp * 10.0f;
        if(val>1.0f) val=1.0f;
        unsigned char c = (unsigned char)(val*255.0f);

        pbo[idx2D].x = c;
        pbo[idx2D].y = c;
        pbo[idx2D].z = c;
        pbo[idx2D].w = 255;
    }
}

// -----------------------------------------------------------------------------
// Morlet wavelet transform in angle
// -----------------------------------------------------------------------------
static inline float morletReal(float t, float s, float w0)
{
    float x = t/s;
    float gauss = expf(-0.5f*x*x);
    return gauss * cosf(w0*x);
}
static inline float morletImag(float t, float s, float w0)
{
    float x = t/s;
    float gauss = expf(-0.5f*x*x);
    return gauss * sinf(w0*x);
}

// We do a naive O(N^2 * numScales) transform
void computeMorletWavelet1D(const std::vector<float>& angleSignal,
                            int N, int numScales,
                            float scaleMin, float scaleMax, float w0,
                            std::vector<float>& waveletScalogram)
{
    for(int sIdx=0; sIdx<numScales; sIdx++)
    {
        float fraction = (float)sIdx/(float)(numScales-1);
        float scale = scaleMin * powf(scaleMax/scaleMin, fraction);

        for(int tau=0; tau<N; tau++)
        {
            float realPart=0.0f, imagPart=0.0f;
            for(int k=0; k<N; k++)
            {
                int idx = (tau + k) % N;
                float x = (float)k;

                float wR = morletReal(x, scale, w0);
                float wI = morletImag(x, scale, w0);

                realPart += angleSignal[idx]*wR;
                imagPart += angleSignal[idx]*wI;
            }
            float mag = sqrtf(realPart*realPart + imagPart*imagPart);
            waveletScalogram[sIdx*N + tau] = mag;
        }
    }
}

// -----------------------------------------------------------------------------
// Sample amplitude around a circle in the slice z=SLICE_Z
// (cx, cy) is the center in that slice
// -----------------------------------------------------------------------------
void sampleAmplitudeCircle(const std::vector<Complex3D>& psiHost,
                           int nx, int ny, int nz, int sliceZ,
                           float cx, float cy, float radius,
                           int numAngles,
                           std::vector<float>& angleSignal)
{
    for(int i=0; i<numAngles; i++)
    {
        float theta = 2.0f*(float)M_PI*(float)i/(float)numAngles;
        float xCoord = cx + radius*cosf(theta);
        float yCoord = cy + radius*sinf(theta);

        int xi = (int)roundf(xCoord);
        int yi = (int)roundf(yCoord);

        // wrap or clamp
        if(xi<0) xi += nx;
        if(yi<0) yi += ny;
        xi = xi % nx;
        yi = yi % ny;

        int idx3D = (sliceZ*ny + yi)*nx + xi;
        float amp = sqrtf(psiHost[idx3D].x*psiHost[idx3D].x + psiHost[idx3D].y*psiHost[idx3D].y);
        angleSignal[i] = amp;
    }
}

// -----------------------------------------------------------------------------
// CUDA-OpenGL interop
// -----------------------------------------------------------------------------
static GLuint pboID;
static struct cudaGraphicsResource* pboResource;

// We'll have TWO wavelet textures: waveletTex1 (for R1) and waveletTex2 (for R2)
static GLuint waveletTex1;
static GLuint waveletTex2;

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

bool createWaveletTexture(GLuint& texID, int width, int height)
{
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void updateWaveletTexture(GLuint texID,
                          const std::vector<float>& waveletData,
                          int numScales, int numAngles)
{
    std::vector<unsigned char> texCPU(numScales*numAngles*4);

    float maxVal = 0.0f;
    for(auto v : waveletData){
        if(v>maxVal) maxVal=v;
    }
    if(maxVal < 1e-9f) maxVal=1e-9f;

    for(int s=0; s<numScales; s++){
        for(int a=0; a<numAngles; a++){
            float val = waveletData[s*numAngles + a]/maxVal;
            if(val>1.0f) val=1.0f;
            unsigned char c = (unsigned char)(val*255.0f);

            int idx = (s*numAngles + a)*4;
            texCPU[idx+0] = c;
            texCPU[idx+1] = c;
            texCPU[idx+2] = c;
            texCPU[idx+3] = 255;
        }
    }

    glBindTexture(GL_TEXTURE_2D, texID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    numAngles, numScales,
                    GL_RGBA, GL_UNSIGNED_BYTE, texCPU.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void drawTexturedQuad(GLuint texID, int x, int y, int w, int h,
                      int windowW, int windowH)
{
    glViewport(x, y, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,1,0,1,-1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texID);

    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2f(0,0);
    glTexCoord2f(1,0); glVertex2f(1,0);
    glTexCoord2f(1,1); glVertex2f(1,1);
    glTexCoord2f(0,1); glVertex2f(0,1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    // We want 3 panels: left PDE slice, center wavelet R1, right wavelet R2
    // We'll make the window 3*NX wide, NY high
    int winW = NX*3;
    int winH = NY;

    GLFWwindow* window = glfwCreateWindow(winW, winH, "3D DVRIPE + Dual Radius Polar Wavelet", nullptr, nullptr);
    if(!window){
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK){
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Allocate 3D field on host
    int n = NX*NY*NZ;
    std::vector<Complex3D> psiHost(n), psiNewHost(n);
    initializeField3D(psiHost, NX, NY, NZ, DX);

    // Allocate on device
    Complex3D* psiDev;
    Complex3D* psiNewDev;
    cudaMalloc(&psiDev,    n*sizeof(Complex3D));
    cudaMalloc(&psiNewDev, n*sizeof(Complex3D));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(Complex3D), cudaMemcpyHostToDevice);

    // Create PBO for PDE slice
    if(!createPBO(NX, NY)){
        std::cerr << "Failed to create PBO\n";
        return -1;
    }

    // Create 2 wavelet textures for R1 and R2
    if(!createWaveletTexture(waveletTex1, NUM_ANGLES, NUM_SCALES)){
        std::cerr << "Failed to create waveletTex1\n";
        return -1;
    }
    if(!createWaveletTexture(waveletTex2, NUM_ANGLES, NUM_SCALES)){
        std::cerr << "Failed to create waveletTex2\n";
        return -1;
    }

    // PDE kernel config
    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((NX+threadsPerBlock.x-1)/threadsPerBlock.x,
                   (NY+threadsPerBlock.y-1)/threadsPerBlock.y,
                   (NZ+threadsPerBlock.z-1)/threadsPerBlock.z);

    // copy kernel config
    int copyThreads=256;
    int copyBlocks=(n+copyThreads-1)/copyThreads;

    // slice kernel config
    dim3 sliceThreads(16,16);
    dim3 sliceBlocks((NX+sliceThreads.x-1)/sliceThreads.x,
                     (NY+sliceThreads.y-1)/sliceThreads.y);

    // Buffers for wavelet
    std::vector<float> angleSignalR1(NUM_ANGLES);
    std::vector<float> angleSignalR2(NUM_ANGLES);

    std::vector<float> waveletScalogramR1(NUM_SCALES*NUM_ANGLES);
    std::vector<float> waveletScalogramR2(NUM_SCALES*NUM_ANGLES);

    while(!glfwWindowShouldClose(window))
    {
        // 1) Evolve PDE
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            pdeUpdateKernel3D<<<numBlocks, threadsPerBlock>>>(psiDev, psiNewDev,
                                                              NX, NY, NZ,
                                                              DX, DT, D, G);
            cudaDeviceSynchronize();
            copyKernel3D<<<copyBlocks, copyThreads>>>(psiDev, psiNewDev, n);
            cudaDeviceSynchronize();
        }

        // 2) Copy to host for wavelet
        cudaMemcpy(psiHost.data(), psiDev, n*sizeof(Complex3D), cudaMemcpyDeviceToHost);

        // 3) Sample amplitude around R1, R2
        sampleAmplitudeCircle(psiHost, NX, NY, NZ, SLICE_Z,
                              NX/2.0f, NY/2.0f, R1, NUM_ANGLES, angleSignalR1);
        sampleAmplitudeCircle(psiHost, NX, NY, NZ, SLICE_Z,
                              NX/2.0f, NY/2.0f, R2, NUM_ANGLES, angleSignalR2);

        // 4) Morlet wavelet in angle for each radius
        computeMorletWavelet1D(angleSignalR1, NUM_ANGLES, NUM_SCALES,
                               1.0f, 60.0f, W0, waveletScalogramR1);
        computeMorletWavelet1D(angleSignalR2, NUM_ANGLES, NUM_SCALES,
                               1.0f, 60.0f, W0, waveletScalogramR2);

        // 5) Update wavelet textures
        updateWaveletTexture(waveletTex1, waveletScalogramR1, NUM_SCALES, NUM_ANGLES);
        updateWaveletTexture(waveletTex2, waveletScalogramR2, NUM_SCALES, NUM_ANGLES);

        // 6) Map PBO + fill slice
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes=0;
        uchar4* d_pbo=nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillSliceKernel<<<sliceBlocks, sliceThreads>>>(d_pbo, psiDev,
                                                       NX, NY, NZ, SLICE_Z);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &pboResource, 0);

        // 7) Render
        glClear(GL_COLOR_BUFFER_BIT);

        // left panel: PDE slice
        {
            glViewport(0, 0, NX, NY);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, NX, 0, NY, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glRasterPos2i(0,0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
            glDrawPixels(NX, NY, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // center panel: wavelet R1
        {
            int panelX = NX; // start at x=NX
            drawTexturedQuad(waveletTex1,
                             panelX, 0,
                             NX, NY,
                             winW, winH);
        }

        // right panel: wavelet R2
        {
            int panelX = NX*2; // start at x=2*NX
            drawTexturedQuad(waveletTex2,
                             panelX, 0,
                             NX, NY,
                             winW, winH);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);
    glDeleteTextures(1, &waveletTex1);
    glDeleteTextures(1, &waveletTex2);

    cudaFree(psiDev);
    cudaFree(psiNewDev);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
