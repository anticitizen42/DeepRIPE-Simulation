/******************************************************************************
 * dvripe_sim_realtime.cu
 *
 * A single-file CUDA + OpenGL + GLFW program that:
 *   1. Simulates a 2D nonlinear Schrödinger-like equation (DVRIPE style).
 *   2. Displays an animated window of the amplitude in real time.
 *
 * Compile on Windows (example):
 *   nvcc dvripe_sim_realtime.cu -o dvripe_sim_realtime.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lglfw3 -lopengl32 -lglew32
 *
 ******************************************************************************/

// --- Windows-specific defines to minimize header bloat ---
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
#endif

// --- Tell GLFW NOT to include gl.h automatically ---
#define GLFW_INCLUDE_NONE

// --- Now include GLFW ---
#include <GLFW/glfw3.h>

// --- Include GLEW after we've prevented GLFW from loading gl.h ---
#include <GL/glew.h>

// Standard headers
#include <iostream>
#include <cmath>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//=============================================================================
// Simulation parameters
//=============================================================================
#define NX 256            // Grid size in x
#define NY 256            // Grid size in y
#define DX 0.1f           // Spatial resolution
#define DT 0.0001f        // Time step
#define D  1.0f           // Diffusion/dispersion coefficient
#define G  1.0f           // Nonlinearity strength

// We'll do ~1 PDE iteration per frame
#define STEPS_PER_FRAME 1

//=============================================================================
// CUDA complex math helpers
//=============================================================================
struct Complex
{
    float x;  // real
    float y;  // imag
};

__device__ inline Complex cAdd(const Complex& a, const Complex& b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__device__ inline Complex cScalarMul(const Complex& a, float s) {
    Complex c;
    c.x = a.x * s;
    c.y = a.y * s;
    return c;
}

__device__ inline Complex cMulI(const Complex& a) {
    // Multiply by i: i*(x + i y) = -y + i x
    Complex c;
    c.x = -a.y;
    c.y =  a.x;
    return c;
}

__device__ inline float cAbs2(const Complex& a) {
    return a.x*a.x + a.y*a.y;
}

//=============================================================================
// CUDA kernels
//=============================================================================

// 1) PDE update kernel: explicit Euler for dψ/dt = iD∇²ψ + iG|ψ|²ψ
__global__ void pdeUpdateKernel(Complex* psi, Complex* psiNew,
                                int nx, int ny, float dx, float dt,
                                float Dval, float Gval)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y index

    if(i < nx && j < ny)
    {
        int idx = j * nx + i;

        // Periodic boundary
        int ip = (i + 1) % nx;
        int im = (i - 1 + nx) % nx;
        int jp = (j + 1) % ny;
        int jm = (j - 1 + ny) % ny;

        int idx_ip = j * nx + ip;
        int idx_im = j * nx + im;
        int idx_jp = jp * nx + i;
        int idx_jm = jm * nx + i;

        // Laplacian
        Complex lap;
        lap.x = (psi[idx_ip].x + psi[idx_im].x + psi[idx_jp].x + psi[idx_jm].x
                 - 4.0f*psi[idx].x) / (dx*dx);
        lap.y = (psi[idx_ip].y + psi[idx_im].y + psi[idx_jp].y + psi[idx_jm].y
                 - 4.0f*psi[idx].y) / (dx*dx);

        // i * D * laplacian
        Complex linear = cMulI(lap);
        linear = cScalarMul(linear, Dval);

        // i * G * |psi|^2 * psi
        float amp2 = cAbs2(psi[idx]);
        Complex nonlinear = cScalarMul(psi[idx], Gval * amp2);
        nonlinear = cMulI(nonlinear);

        // dpsi/dt
        Complex dpsi_dt = cAdd(linear, nonlinear);

        // Euler step
        psiNew[idx].x = psi[idx].x + dt * dpsi_dt.x;
        psiNew[idx].y = psi[idx].y + dt * dpsi_dt.y;
    }
}

// 2) Copy kernel: psi = psiNew
__global__ void copyKernel(Complex* psi, Complex* psiNew, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        psi[idx] = psiNew[idx];
    }
}

// 3) Color mapping kernel: fill PBO with RGBA based on amplitude
//    amplitude = sqrt(psi.x^2 + psi.y^2).
__global__ void fillPBOKernel(uchar4* pbo, const Complex* psi, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y index

    if(i < nx && j < ny)
    {
        int idx = j * nx + i;
        float amp = sqrtf(psi[idx].x * psi[idx].x + psi[idx].y * psi[idx].y);

        // Simple color mapping: grayscale from 0 (black) to 1 (white)
        float val = amp * 10.0f; // scale factor
        if(val > 1.0f) val = 1.0f;  // clamp

        unsigned char c = static_cast<unsigned char>(val * 255.0f);

        // Write to pbo (RGBA)
        pbo[idx].x = c;   // R
        pbo[idx].y = c;   // G
        pbo[idx].z = c;   // B
        pbo[idx].w = 255; // A
    }
}

//=============================================================================
// Host code for field initialization
//=============================================================================
void initializeField(std::vector<Complex>& psi, int nx, int ny, float dx)
{
    float cx = nx / 2.0f;
    float cy = ny / 2.0f;

    for(int j=0; j<ny; j++)
    {
        for(int i=0; i<nx; i++)
        {
            int idx = j*nx + i;
            float x = i - cx;
            float y = j - cy;
            float r = sqrtf(x*x + y*y);
            float theta = atan2f(y, x);

            // Gaussian amplitude around center
            float amplitude = expf(-r*r / (nx * dx * 0.1f));
            // Single vortex phase
            float phase = theta;

            psi[idx].x = amplitude * cosf(phase);
            psi[idx].y = amplitude * sinf(phase);
        }
    }
}

//=============================================================================
// Global variables for CUDA-OpenGL interop
//=============================================================================
static GLuint pboID;  // OpenGL pixel buffer object
static struct cudaGraphicsResource* cudaPboResource; // CUDA resource

//=============================================================================
// Create PBO, register with CUDA
//=============================================================================
bool createPBO(int width, int height)
{
    // Create pixel buffer object
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register PBO with CUDA
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cudaPboResource, pboID, cudaGraphicsRegisterFlagsWriteDiscard
    );
    if(err != cudaSuccess)
    {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: "
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

//=============================================================================
// Render PBO to screen using glDrawPixels
//=============================================================================
void renderPBO(int width, int height)
{
    glRasterPos2i(-1, -1); // lower-left corner
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

//=============================================================================
// Main
//=============================================================================
int main()
{
    //--------------------------------------------------------------------------
    // Initialize GLFW and create window
    //--------------------------------------------------------------------------
    if(!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(NX, NY, "DVRIPE Real-Time Simulation", nullptr, nullptr);
    if(!window)
    {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    //--------------------------------------------------------------------------
    // Initialize GLEW AFTER creating an OpenGL context
    //--------------------------------------------------------------------------
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK)
    {
        std::cerr << "GLEW Error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    //--------------------------------------------------------------------------
    // Allocate and initialize the field on host
    //--------------------------------------------------------------------------
    int n = NX * NY;
    std::vector<Complex> psiHost(n), psiNewHost(n);
    initializeField(psiHost, NX, NY, DX);

    //--------------------------------------------------------------------------
    // Allocate on device
    //--------------------------------------------------------------------------
    Complex *psiDev, *psiNewDev;
    cudaMalloc(&psiDev, n*sizeof(Complex));
    cudaMalloc(&psiNewDev, n*sizeof(Complex));

    // Copy initial data to device
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(Complex), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // Create a Pixel Buffer Object (PBO) and register it with CUDA
    //--------------------------------------------------------------------------
    if(!createPBO(NX, NY))
    {
        std::cerr << "Failed to create/register PBO!" << std::endl;
        return -1;
    }

    //--------------------------------------------------------------------------
    // Prepare CUDA kernel configuration
    //--------------------------------------------------------------------------
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks( (NX + threadsPerBlock.x - 1)/threadsPerBlock.x,
                    (NY + threadsPerBlock.y - 1)/threadsPerBlock.y );

    //--------------------------------------------------------------------------
    // Main loop
    //--------------------------------------------------------------------------
    while(!glfwWindowShouldClose(window))
    {
        // 1) Run PDE updates
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // PDE update
            pdeUpdateKernel<<<numBlocks, threadsPerBlock>>>(psiDev, psiNewDev,
                                                            NX, NY, DX, DT, D, G);
            cudaDeviceSynchronize();

            // Copy psiNewDev -> psiDev
            int threads = 256;
            int blocks = (n + threads - 1)/threads;
            copyKernel<<<blocks, threads>>>(psiDev, psiNewDev, n);
            cudaDeviceSynchronize();
        }

        // 2) Map the PBO and fill with color data
        cudaGraphicsMapResources(1, &cudaPboResource, 0);
        size_t numBytes = 0;
        uchar4* d_pbo = nullptr;
        cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, cudaPboResource);

        fillPBOKernel<<<numBlocks, threadsPerBlock>>>(d_pbo, psiDev, NX, NY);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &cudaPboResource, 0);

        // 3) Clear screen and render
        glClear(GL_COLOR_BUFFER_BIT);
        renderPBO(NX, NY);

        // 4) Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //--------------------------------------------------------------------------
    // Cleanup
    //--------------------------------------------------------------------------
    cudaGraphicsUnregisterResource(cudaPboResource);
    glDeleteBuffers(1, &pboID);

    cudaFree(psiDev);
    cudaFree(psiNewDev);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
