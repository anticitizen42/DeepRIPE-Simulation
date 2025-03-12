/******************************************************************************
 * dvripe_sim_wavelet_realtime.cu
 *
 * Single-file CUDA + OpenGL + GLFW demonstration:
 *   1) Simulates a 2D nonlinear Schrödinger-like PDE (DVRIPE style).
 *   2) Samples amplitude around a circular path each frame.
 *   3) Computes a naive 1D Morlet wavelet transform on the CPU.
 *   4) Renders two panels in one window:
 *      - Left: PDE amplitude (grayscale).
 *      - Right: wavelet scalogram (scale vs. angle).
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_wavelet_realtime.cu -o dvripe_sim_wavelet_realtime.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lglew32 -lglfw3 -lopengl32
 *
 ******************************************************************************/

// ------------------------------
// 1) Includes and Definitions
// ------------------------------
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Define _USE_MATH_DEFINES before <cmath> so M_PI is recognized on MSVC
#define _USE_MATH_DEFINES
#include <cmath>

// If M_PI is still not defined, define it manually
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Prevent GLFW from including any GL headers
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Include GLEW AFTER GLFW
#include <GL/glew.h>

// CUDA-OpenGL interop
#include <cuda_gl_interop.h>

// ------------------------------
// Simulation parameters
// ------------------------------
static const int    NX = 256;     // Grid size in x
static const int    NY = 256;     // Grid size in y
static const float  DX = 0.1f;    // Spatial resolution
static const float  DT = 0.0001f; // Time step
static const float  D  = 1.0f;    // Diffusion/dispersion coefficient
static const float  G  = 1.0f;    // Nonlinearity strength

static const int    STEPS_PER_FRAME = 1; // PDE steps per rendered frame

// Wavelet/scalogram
static const int    NUM_ANGLES = 360; // sample points around circle
static const float  RADIUS     = 10.0f;
static const int    NUM_SCALES = 64;  // wavelet scales
static const float  W0         = 6.0f; // Morlet carrier frequency

// ------------------------------
// CUDA complex struct
// ------------------------------
struct Complex {
    float x; // real
    float y; // imag
};

__device__ inline float cAbs2(const Complex& a) {
    return a.x*a.x + a.y*a.y;
}

// ------------------------------
// PDE Update Kernels
// ------------------------------

// 1) PDE update: dψ/dt = iD∇²ψ + iG|ψ|²ψ
__global__ void pdeUpdateKernel(Complex* psi, Complex* psiNew,
                                int nx, int ny, float dx, float dt,
                                float Dval, float Gval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y index
    if(i < nx && j < ny)
    {
        int idx = j*nx + i;

        // Periodic boundary
        int ip = (i + 1) % nx;
        int im = (i - 1 + nx) % nx;
        int jp = (j + 1) % ny;
        int jm = (j - 1 + ny) % ny;

        int idx_ip = j*nx + ip;
        int idx_im = j*nx + im;
        int idx_jp = jp*nx + i;
        int idx_jm = jm*nx + i;

        // Laplacian
        Complex lap;
        lap.x = (psi[idx_ip].x + psi[idx_im].x + psi[idx_jp].x + psi[idx_jm].x
                 - 4.0f*psi[idx].x) / (dx*dx);
        lap.y = (psi[idx_ip].y + psi[idx_im].y + psi[idx_jp].y + psi[idx_jm].y
                 - 4.0f*psi[idx].y) / (dx*dx);

        // i * D * laplacian
        // multiply lap by i => (-lap.y, lap.x)
        Complex linear;
        linear.x = -lap.y * Dval;
        linear.y =  lap.x * Dval;

        // i * G * |psi|^2 * psi
        float amp2 = cAbs2(psi[idx]);
        // multiply psi by i => (-psi[idx].y, psi[idx].x)
        Complex nonlin;
        nonlin.x = -psi[idx].y;
        nonlin.y =  psi[idx].x;
        // scale by Gval * amp2
        nonlin.x *= (Gval * amp2);
        nonlin.y *= (Gval * amp2);

        // dpsi/dt
        Complex dpsi_dt;
        dpsi_dt.x = linear.x + nonlin.x;
        dpsi_dt.y = linear.y + nonlin.y;

        // Euler step
        psiNew[idx].x = psi[idx].x + dt*dpsi_dt.x;
        psiNew[idx].y = psi[idx].y + dt*dpsi_dt.y;
    }
}

// 2) Copy kernel
__global__ void copyKernel(Complex* psi, Complex* psiNew, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        psi[idx] = psiNew[idx];
    }
}

// ------------------------------
// Field Initialization
// ------------------------------
void initializeField(std::vector<Complex>& psiHost, int nx, int ny, float dx)
{
    float cx = nx/2.0f;
    float cy = ny/2.0f;
    for(int j=0; j<ny; j++)
    {
        for(int i=0; i<nx; i++)
        {
            int idx = j*nx + i;
            float x = i - cx;
            float y = j - cy;
            float r = std::sqrt(x*x + y*y);
            float theta = std::atan2(y, x);

            // Gaussian amplitude
            float amplitude = std::exp(-r*r / (nx*dx*0.1f));
            // Single-charged vortex
            float phase = theta;

            psiHost[idx].x = amplitude * std::cos(phase);
            psiHost[idx].y = amplitude * std::sin(phase);
        }
    }
}

// ------------------------------
// PDE Amplitude -> Grayscale
// ------------------------------
__global__ void fillPBOKernel(uchar4* pbo, const Complex* psi, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < nx && j < ny)
    {
        int idx = j*nx + i;
        float amp = std::sqrt(psi[idx].x*psi[idx].x + psi[idx].y*psi[idx].y);

        // map amplitude to grayscale
        float val = amp * 10.0f; // arbitrary scaling
        if(val > 1.0f) val = 1.0f;
        unsigned char c = static_cast<unsigned char>(val * 255.0f);

        pbo[idx].x = c;
        pbo[idx].y = c;
        pbo[idx].z = c;
        pbo[idx].w = 255;
    }
}

// ------------------------------
// CPU Morlet Wavelet Transform
// ------------------------------
static inline float morletReal(float t, float s, float w0)
{
    float x = t/s;
    float gauss = std::exp(-0.5f*x*x);
    return gauss * std::cos(w0*x);
}
static inline float morletImag(float t, float s, float w0)
{
    float x = t/s;
    float gauss = std::exp(-0.5f*x*x);
    return gauss * std::sin(w0*x);
}

void computeMorletWavelet1D(const std::vector<float>& angleSignal,
                            int N, int numScales,
                            float scaleMin, float scaleMax,
                            float w0,
                            std::vector<float>& waveletScalogram)
{
    // waveletScalogram.size() == numScales*N
    for(int sIdx=0; sIdx<numScales; sIdx++)
    {
        // geometric spacing
        float fraction = (float)sIdx / (float)(numScales - 1);
        float scale = scaleMin * std::pow(scaleMax/scaleMin, fraction);

        for(int tau=0; tau<N; tau++)
        {
            float realPart = 0.0f;
            float imagPart = 0.0f;
            // naive convolution
            for(int k=0; k<N; k++)
            {
                int idx = (tau + k) % N;
                float x = (float)k;

                float wR = morletReal(x, scale, w0);
                float wI = morletImag(x, scale, w0);

                realPart += angleSignal[idx]*wR;
                imagPart += angleSignal[idx]*wI;
            }
            float mag = std::sqrt(realPart*realPart + imagPart*imagPart);
            waveletScalogram[sIdx*N + tau] = mag;
        }
    }
}

// ------------------------------
// Sample amplitude around circle
// ------------------------------
void sampleAmplitudeCircle(const std::vector<Complex>& psiHost,
                           int nx, int ny,
                           float cx, float cy, float radius,
                           int numAngles,
                           std::vector<float>& angleSignal)
{
    for(int i=0; i<numAngles; i++)
    {
        float theta = 2.0f * (float)M_PI * (float)i / (float)numAngles;
        float xCoord = cx + radius*std::cos(theta);
        float yCoord = cy + radius*std::sin(theta);

        int xi = (int)std::round(xCoord) % nx;
        int yi = (int)std::round(yCoord) % ny;
        if(xi < 0) xi += nx; // wrap
        if(yi < 0) yi += ny;

        int idx = yi*nx + xi;
        float amp = std::sqrt(psiHost[idx].x*psiHost[idx].x + psiHost[idx].y*psiHost[idx].y);
        angleSignal[i] = amp;
    }
}

// ------------------------------
// CUDA-OpenGL Interop
// ------------------------------
static GLuint pboID;
static struct cudaGraphicsResource* pboResource;
static GLuint waveletTexID;

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

bool createWaveletTexture(int width, int height)
{
    glGenTextures(1, &waveletTexID);
    glBindTexture(GL_TEXTURE_2D, waveletTexID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

void updateWaveletTexture(const std::vector<float>& waveletData,
                          int numScales, int numAngles)
{
    // waveletData.size() == numScales * numAngles
    // Convert to RGBA8
    std::vector<unsigned char> texCPU(numScales*numAngles*4);

    // find max
    float maxVal = 0.0f;
    for(size_t i=0; i<waveletData.size(); i++) {
        if(waveletData[i] > maxVal) maxVal = waveletData[i];
    }
    if(maxVal < 1e-9f) maxVal = 1e-9f;

    for(int s=0; s<numScales; s++)
    {
        for(int a=0; a<numAngles; a++)
        {
            float val = waveletData[s*numAngles + a] / maxVal;
            if(val>1.0f) val=1.0f;
            unsigned char c = static_cast<unsigned char>(val*255.0f);

            int idx = (s*numAngles + a)*4;
            texCPU[idx+0] = c;
            texCPU[idx+1] = c;
            texCPU[idx+2] = c;
            texCPU[idx+3] = 255;
        }
    }

    // Upload
    glBindTexture(GL_TEXTURE_2D, waveletTexID);
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

// ------------------------------
// Main
// ------------------------------
int main()
{
    if(!glfwInit()) {
        std::cerr << "Failed to init GLFW!\n";
        return -1;
    }

    int winW = NX*2; // double width
    int winH = NY;

    GLFWwindow* window = glfwCreateWindow(winW, winH, "DVRIPE + Wavelet Realtime", nullptr, nullptr);
    if(!window) {
        std::cerr << "Failed to create GLFW window!\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK) {
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Allocate field on host
    int n = NX*NY;
    std::vector<Complex> psiHost(n), psiNewHost(n);
    initializeField(psiHost, NX, NY, DX);

    // Allocate on device
    Complex *psiDev, *psiNewDev;
    cudaMalloc(&psiDev,    n*sizeof(Complex));
    cudaMalloc(&psiNewDev, n*sizeof(Complex));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(Complex), cudaMemcpyHostToDevice);

    // Create PBO for PDE amplitude
    if(!createPBO(NX, NY)) {
        std::cerr << "Failed to create PBO!\n";
        return -1;
    }

    // Create texture for wavelet scalogram (NUM_ANGLES x NUM_SCALES)
    if(!createWaveletTexture(NUM_ANGLES, NUM_SCALES)) {
        std::cerr << "Failed to create wavelet texture!\n";
        return -1;
    }

    // Prepare CUDA kernel config
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((NX+threadsPerBlock.x-1)/threadsPerBlock.x,
                   (NY+threadsPerBlock.y-1)/threadsPerBlock.y);
    int copyThreads = 256;
    int copyBlocks = (n + copyThreads -1)/copyThreads;

    // For wavelet
    std::vector<float> angleSignal(NUM_ANGLES);
    std::vector<float> waveletScalogram(NUM_SCALES*NUM_ANGLES);

    while(!glfwWindowShouldClose(window))
    {
        // 1) Evolve PDE
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            pdeUpdateKernel<<<numBlocks, threadsPerBlock>>>(psiDev, psiNewDev, NX, NY, DX, DT, D, G);
            cudaDeviceSynchronize();
            copyKernel<<<copyBlocks, copyThreads>>>(psiDev, psiNewDev, n);
            cudaDeviceSynchronize();
        }

        // 2) Copy back to host for wavelet
        cudaMemcpy(psiHost.data(), psiDev, n*sizeof(Complex), cudaMemcpyDeviceToHost);

        // 3) Sample amplitude around circle
        sampleAmplitudeCircle(psiHost, NX, NY,
                              NX/2.0f, NY/2.0f,
                              RADIUS, NUM_ANGLES,
                              angleSignal);

        // 4) Morlet wavelet transform
        //    We'll do scaleMin=1, scaleMax=80 as an example
        computeMorletWavelet1D(angleSignal, NUM_ANGLES, NUM_SCALES,
                               1.0f, 80.0f, W0,
                               waveletScalogram);

        // 5) Update wavelet texture
        updateWaveletTexture(waveletScalogram, NUM_SCALES, NUM_ANGLES);

        // 6) Map PBO -> fill with PDE amplitude
        cudaGraphicsMapResources(1, &pboResource, 0);
        size_t numBytes = 0;
        uchar4* d_pbo = nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboResource);

        fillPBOKernel<<<numBlocks, threadsPerBlock>>>(d_pbo, psiDev, NX, NY);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &pboResource, 0);

        // 7) Render
        glClear(GL_COLOR_BUFFER_BIT);

        // Left half: PDE amplitude
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

        // Right half: wavelet scalogram
        {
            // waveletTexID is (NUM_ANGLES x NUM_SCALES)
            drawTexturedQuad(waveletTexID,
                             NX, 0,  // x, y in window
                             NX, NY, // width, height
                             winW, winH);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(pboResource);
    glDeleteBuffers(1, &pboID);
    glDeleteTextures(1, &waveletTexID);

    cudaFree(psiDev);
    cudaFree(psiNewDev);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
