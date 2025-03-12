/******************************************************************************
 * dvripe_sim_3d_ssf_phase_swirl.cu
 *
 * A single-file CUDA + cuFFT + OpenGL + GLFW program that:
 *   1) Runs a 3D Split-Step Fourier simulation of the focusing Schrödinger PDE
 *   2) Renders a single slice (z=NZ/2) of amplitude in grayscale
 *   3) Detects phase vortex swirl in that slice and recenters the volume so the
 *      vortex remains at the slice center
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_ssf_phase_swirl.cu -o dvripe_sim_3d_ssf_phase_swirl.exe ^
        -I"C:\vcpkg\installed\x64-windows\include" ^
        -L"C:\vcpkg\installed\x64-windows\lib" ^
        -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_ssf_phase_swirl.cu -o dvripe_sim_3d_ssf_phase_swirl \
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

// Prevent GLFW from including gl.h
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

//---------------------------------------------------------------------------
// Simulation grid and PDE parameters
//---------------------------------------------------------------------------
static const int   NX   = 64;       // domain size in x
static const int   NY   = 64;       // domain size in y
static const int   NZ   = 64;       // domain size in z
static const float DX   = 0.1f;     // spatial step
static const float DT   = 0.0001f;  // time step
static const float DVAL = 1.0f;     // dispersion coefficient
static const float GVAL = 1.0f;     // focusing strength
static const int   STEPS_PER_FRAME = 1; // PDE steps each rendered frame

//---------------------------------------------------------------------------
// GPU float2 helpers
//---------------------------------------------------------------------------
__device__ inline float2 cMul(const float2& a, const float2& b)
{
    // complex multiply: (a.x + i a.y)*(b.x + i b.y)
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
// PDE kernels: 3D focusing Schrödinger with split-step
//---------------------------------------------------------------------------
__global__ void nonlinearHalfStepKernel(float2* psi, int n, float dt, float G)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float2 val = psi[idx];
        float amp2 = cAbs2(val);
        float phase = G * amp2 * (dt*0.5f);
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

        // wave numbers with shift
        int sx = (x < nx/2) ? x : (x - nx);
        int sy = (y < ny/2) ? y : (y - ny);
        int sz = (z < nz/2) ? z : (z - nz);

        float kx = sx * dkx;
        float ky = sy * dky;
        float kz = sz * dkz;
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
// Compute amplitude in the entire 3D domain
//---------------------------------------------------------------------------
__global__ void computeAmplitudeKernel3D(const float2* psi,
                                         float* amplitude,
                                         int nx, int ny, int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < nx*ny*nz)
    {
        float2 val = psi[idx];
        float amp = sqrtf(val.x*val.x + val.y*val.y);
        amplitude[idx] = amp;
    }
}

//---------------------------------------------------------------------------
// Extract the phase from a single slice z=NZ/2
//---------------------------------------------------------------------------
__global__ void slicePhaseKernel(const float2* psi,
                                 float* phaseOut,
                                 int nx, int ny, int nz, int sliceZ)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<nx && j<ny)
    {
        int idx3D = (sliceZ*ny + j)*nx + i;
        float2 val = psi[idx3D];
        float ph = atan2f(val.y, val.x);
        int idx2D = j*nx + i;
        phaseOut[idx2D] = ph;
    }
}

//---------------------------------------------------------------------------
// CPU swirl detection: find cell with ~ ±2π phase swirl
//---------------------------------------------------------------------------
static float unwrapPhaseDiff(float p1, float p2)
{
    float diff = p2 - p1;
    while(diff >  M_PI) diff -= 2.0f*M_PI;
    while(diff < -M_PI) diff += 2.0f*M_PI;
    return diff;
}

static void findVortexByPhaseSwirl(const std::vector<float>& phaseSlice,
                                   int nx, int ny,
                                   int& vortexX, int& vortexY)
{
    float maxSwirlMag = 0.0f;
    vortexX = nx/2;
    vortexY = ny/2;

    for(int j=0; j<ny-1; j++)
    {
        for(int i=0; i<nx-1; i++)
        {
            float p00 = phaseSlice[j*nx + i];
            float p10 = phaseSlice[j*nx + (i+1)];
            float p01 = phaseSlice[(j+1)*nx + i];
            float p11 = phaseSlice[(j+1)*nx + (i+1)];

            float d1 = unwrapPhaseDiff(p00, p10);
            float d2 = unwrapPhaseDiff(p10, p11);
            float d3 = unwrapPhaseDiff(p11, p01);
            float d4 = unwrapPhaseDiff(p01, p00);
            float swirl = d1 + d2 + d3 + d4; // ~ ±2π for a vortex

            float mag = fabsf(swirl);
            if(mag > maxSwirlMag)
            {
                maxSwirlMag = mag;
                vortexX = i;
                vortexY = j;
            }
        }
    }
}

//---------------------------------------------------------------------------
// Minimal shaders for a single slice at z=0.5
//---------------------------------------------------------------------------
static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 inPos;
out vec3 fragPos;
uniform mat4 uMVP;
void main()
{
    fragPos = inPos;
    gl_Position = uMVP * vec4(inPos, 1.0);
}
)";

static const char* fragmentShaderSource = R"(
#version 330 core
in vec3 fragPos;
out vec4 outColor;
uniform sampler3D uVolume;

void main()
{
    float xCoord = fragPos.x + 0.5;
    float yCoord = fragPos.y + 0.5;
    if(xCoord < 0.0 || xCoord > 1.0 || yCoord < 0.0 || yCoord > 1.0)
    {
        outColor = vec4(0,0,0,1);
        return;
    }
    float zCoord = 0.5;
    float amp = texture(uVolume, vec3(xCoord,yCoord,zCoord)).r;
    outColor = vec4(amp, amp, amp, 1.0);
}
)";

//---------------------------------------------------------------------------
// OpenGL geometry: a simple cube in [-0.5..+0.5]^3
//---------------------------------------------------------------------------
static GLuint cubeVAO = 0;
static GLuint cubeVBO = 0;

static void createCube()
{
    float vertices[] =
    {
        // 36 vertices of a cube
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,

         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,

         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

        -0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f, -0.5f,

         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f,  0.5f,

        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,

         0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f, -0.5f, -0.5f
    };

    glGenVertexArrays(1, &cubeVAO);
    glBindVertexArray(cubeVAO);

    glGenBuffers(1, &cubeVBO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          3*sizeof(float), (void*)0);

    glBindVertexArray(0);
}

//---------------------------------------------------------------------------
// Shader compilation
//---------------------------------------------------------------------------
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
    }
    return shader;
}

//---------------------------------------------------------------------------
// We'll store amplitude in a 3D texture for the slice display
//---------------------------------------------------------------------------
static GLuint amplitudeTex = 0;
static GLuint prog         = 0;

static GLuint createShaderProgram()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER,   vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint sp = glCreateProgram();
    glAttachShader(sp, vs);
    glAttachShader(sp, fs);
    glLinkProgram(sp);

    GLint success = 0;
    glGetProgramiv(sp, GL_LINK_STATUS, &success);
    if(!success)
    {
        char log[1024];
        glGetProgramInfoLog(sp, 1024, nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return sp;
}

static void upload3DTexture(const float* data,
                            int nx, int ny, int nz)
{
    glBindTexture(GL_TEXTURE_3D, amplitudeTex);
    glTexSubImage3D(GL_TEXTURE_3D, 0,
                    0, 0, 0, nx, ny, nz,
                    GL_RED, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_3D, 0);
}

//---------------------------------------------------------------------------
// Main
//---------------------------------------------------------------------------
int main()
{
    // Initialize GLFW
    if(!glfwInit())
    {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    int winW = 800;
    int winH = 600;
    GLFWwindow* window = glfwCreateWindow(winW, winH,
                                          "DVRIPE 3D Phase Swirl Tracking",
                                          nullptr, nullptr);
    if(!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK)
    {
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Allocate 3D field on host
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);

    // Initial swirl
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
                float amp = expf(-r2/(NX*DX*0.3f));
                float th  = atan2f(dy_, dx_);
                psiHost[idx] = make_float2(amp*cosf(th), amp*sinf(th));
            }
        }
    }

    // Allocate on device
    float2* psiDev = nullptr;
    cudaMalloc(&psiDev, n*sizeof(float2));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(float2),
               cudaMemcpyHostToDevice);

    // Create cuFFT plan
    cufftHandle plan;
    if(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C) != CUFFT_SUCCESS)
    {
        std::cerr << "cufftPlan3d failed!\n";
        return -1;
    }

    float dkx = 2.0f*M_PI / (NX*DX);
    float dky = 2.0f*M_PI / (NY*DX);
    float dkz = 2.0f*M_PI / (NZ*DX);

    // For amplitude
    float* ampDev = nullptr;
    cudaMalloc(&ampDev, n*sizeof(float));

    // For phase slice
    float* phaseSliceDev = nullptr;
    cudaMalloc(&phaseSliceDev, NX*NY*sizeof(float));

    // Create 3D texture for amplitude
    glGenTextures(1, &amplitudeTex);
    glBindTexture(GL_TEXTURE_3D, amplitudeTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F,
                 NX, NY, NZ, 0,
                 GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    // Create cube geometry and shader
    createCube();
    prog = createShaderProgram();

    glUseProgram(prog);
    GLint locVol = glGetUniformLocation(prog, "uVolume");
    glUniform1i(locVol, 0);
    glUseProgram(0);

    // PDE kernel config
    int blockSize = 256;
    int gridSize  = (n + blockSize -1)/blockSize;

    dim3 threads2D(16,16);
    dim3 blocks2D((NX+threads2D.x-1)/threads2D.x,
                  (NY+threads2D.y-1)/threads2D.y);

    // Disable depth test so we can see the slice
    glDisable(GL_DEPTH_TEST);

    // We'll orbit the camera around y-axis
    static float angleY = 0.0f;

    // The main loop
    while(!glfwWindowShouldClose(window))
    {
        // 1) PDE Steps
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // Nonlinear half-step
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();

            // Forward FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev,
                                 (cufftComplex*)psiDev, CUFFT_FORWARD);

            // Linear step
            linearStepKernel<<<gridSize, blockSize>>>(psiDev,
                                                      NX, NY, NZ,
                                                      DT, DVAL,
                                                      dkx, dky, dkz);
            cudaDeviceSynchronize();

            // Inverse FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev,
                                 (cufftComplex*)psiDev, CUFFT_INVERSE);

            // Scale
            float scaleVal = 1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<gridSize, blockSize>>>(psiDev, scaleVal, n);
            cudaDeviceSynchronize();

            // Nonlinear half-step
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();
        }

        // 2) Compute amplitude for entire 3D domain -> upload to 3D texture
        computeAmplitudeKernel3D<<<gridSize, blockSize>>>(psiDev, ampDev, NX, NY, NZ);
        cudaDeviceSynchronize();

        std::vector<float> ampHost(n);
        cudaMemcpy(ampHost.data(), ampDev, n*sizeof(float), cudaMemcpyDeviceToHost);
        upload3DTexture(ampHost.data(), NX, NY, NZ);

        // 3) Extract phase in slice z=NZ/2
        int sliceZ = NZ/2;
        dim3 blocks2Dphase = blocks2D; // same config
        slicePhaseKernel<<<blocks2Dphase, threads2D>>>(psiDev,
                                                       phaseSliceDev,
                                                       NX, NY, NZ,
                                                       sliceZ);
        cudaDeviceSynchronize();

        // Copy slice phase to host
        std::vector<float> phaseHost(NX*NY);
        cudaMemcpy(phaseHost.data(), phaseSliceDev, NX*NY*sizeof(float),
                   cudaMemcpyDeviceToHost);

        // 4) Swirl detection
        int bestX = NX/2;
        int bestY = NY/2;
        {
            float maxSwirlMag = 0.0f;
            for(int j=0; j<NY-1; j++)
            {
                for(int i=0; i<NX-1; i++)
                {
                    float p00 = phaseHost[j*NX + i];
                    float p10 = phaseHost[j*NX + (i+1)];
                    float p01 = phaseHost[(j+1)*NX + i];
                    float p11 = phaseHost[(j+1)*NX + (i+1)];

                    // unwrap
                    auto unwrapPhaseDiff = [&](float p1, float p2)
                    {
                        float diff = p2 - p1;
                        while(diff >  M_PI) diff -= 2.0f*M_PI;
                        while(diff < -M_PI) diff += 2.0f*M_PI;
                        return diff;
                    };

                    float d1 = unwrapPhaseDiff(p00, p10);
                    float d2 = unwrapPhaseDiff(p10, p11);
                    float d3 = unwrapPhaseDiff(p11, p01);
                    float d4 = unwrapPhaseDiff(p01, p00);
                    float swirl = d1 + d2 + d3 + d4; // ~ ±2π

                    float mag = fabsf(swirl);
                    if(mag > maxSwirlMag)
                    {
                        maxSwirlMag = mag;
                        bestX = i;
                        bestY = j;
                    }
                }
            }
        }

        // 5) Recenter domain so (bestX, bestY) maps to (0.5, 0.5)
        float fx = (bestX + 0.5f)/(float)NX;
        float fy = (bestY + 0.5f)/(float)NY;
        float fz = 0.5f;
        float shiftX = 0.5f - fx;
        float shiftY = 0.5f - fy;
        float shiftZ = 0.5f - fz;

        // 6) Orbit camera
        angleY += 0.01f;

        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);

        // Basic perspective
        float fov   = 60.0f*(M_PI/180.0f);
        float aspect= (float)winW/(float)winH;
        float nearP = 0.1f;
        float farP  = 10.0f;
        std::vector<float> matP(16,0.0f);

        float f = 1.0f / tanf(fov*0.5f);
        matP[0] = f/aspect;
        matP[5] = f;
        matP[10] = (farP + nearP)/(nearP - farP);
        matP[11] = -1.0f;
        matP[14] = (2.0f*farP*nearP)/(nearP - farP);

        // Minimal orbit camera around y-axis
        float dist = 1.5f;
        float eyex = dist*sinf(angleY);
        float eyey = 0.0f;
        float eyez = dist*cosf(angleY);

        std::vector<float> eye{eyex, eyey, eyez};
        std::vector<float> center{0.0f, 0.0f, 0.0f};
        std::vector<float> up{0.0f, 1.0f, 0.0f};

        auto sub3 = [&](float x1, float y1, float z1,
                        float x2, float y2, float z2)
        {
            return std::vector<float>{x1-x2, y1-y2, z1-z2};
        };
        auto norm3 = [&](const std::vector<float>& v)
        {
            float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            return std::vector<float>{v[0]/len, v[1]/len, v[2]/len};
        };
        auto cross3 = [&](const std::vector<float>& a,
                          const std::vector<float>& b)
        {
            return std::vector<float>{
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            };
        };

        // Build view matrix
        auto fwd = norm3(sub3(center[0], center[1], center[2],
                              eye[0], eye[1], eye[2]));
        auto rht = norm3(cross3(fwd, up));
        auto u2  = cross3(rht, fwd);

        std::vector<float> matV(16,0.0f);
        matV[0]  = rht[0];
        matV[1]  = u2[0];
        matV[2]  = -fwd[0];
        matV[4]  = rht[1];
        matV[5]  = u2[1];
        matV[6]  = -fwd[1];
        matV[8]  = rht[2];
        matV[9]  = u2[2];
        matV[10] = -fwd[2];
        matV[15] = 1.0f;

        float tx = -(rht[0]*eye[0] + rht[1]*eye[1] + rht[2]*eye[2]);
        float ty = -(u2[0]*eye[0]  + u2[1]*eye[1]  + u2[2]*eye[2]);
        float tz =  (fwd[0]*eye[0] + fwd[1]*eye[1] + fwd[2]*eye[2]);
        matV[12] = tx;
        matV[13] = ty;
        matV[14] = tz;

        // Model transform => shift domain by (shiftX, shiftY, shiftZ)
        auto matIdentity4 = [&]()
        {
            std::vector<float> M(16,0.0f);
            M[0]=1.0f; M[5]=1.0f; M[10]=1.0f; M[15]=1.0f;
            return M;
        };
        auto matTranslate = [&](float sx, float sy, float sz)
        {
            auto T = matIdentity4();
            T[12] = sx;
            T[13] = sy;
            T[14] = sz;
            return T;
        };
        auto matMultiply = [&](const std::vector<float>& A,
                               const std::vector<float>& B)
        {
            std::vector<float> M(16,0.0f);
            for(int r=0; r<4; r++)
            {
                for(int c=0; c<4; c++)
                {
                    for(int k=0; k<4; k++)
                    {
                        M[r*4 + c] += A[r*4 + k]*B[k*4 + c];
                    }
                }
            }
            return M;
        };

        auto matM  = matTranslate(shiftX, shiftY, shiftZ);
        auto matVM = matMultiply(matV, matM);
        auto matPVM= matMultiply(matP, matVM);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(prog);

        GLint locMVP = glGetUniformLocation(prog, "uMVP");
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, matPVM.data());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, amplitudeTex);

        glBindVertexArray(cubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cufftDestroy(plan);
    cudaFree(psiDev);
    cudaFree(ampDev);
    cudaFree(phaseSliceDev);

    glDeleteTextures(1, &amplitudeTex);
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
