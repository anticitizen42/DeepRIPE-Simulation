/******************************************************************************
 * dvripe_sim_3d_ssf_follow_vortex.cu
 *
 * Demonstrates a 3D Split-Step Fourier integrator for a focusing Schrödinger 
 * equation, then locates the maximum amplitude ("vortex center") each frame and 
 * translates the volume so that point is at the center of the view. We sample 
 * only one slice (z=0.5) in texture space for simplicity.
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_ssf_follow_vortex.cu -o dvripe_sim_3d_ssf_follow_vortex.exe ^
        -I"C:\vcpkg\installed\x64-windows\include" ^
        -L"C:\vcpkg\installed\x64-windows\lib" ^
        -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_ssf_follow_vortex.cu -o dvripe_sim_3d_ssf_follow_vortex \
 *       -lcufft -lGLEW -lglfw -lGL
 *
 * Notes:
 *   - Domain is 64³, G=0.2, DT=0.0005 => moderate focusing. 
 *   - We do a single-slice visualization (z=0.5) to keep the code simpler. 
 *   - We find the maximum amplitude index each frame, and shift the model 
 *     matrix so that point appears at the center in [-0.5..0.5].
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

// OpenGL / GLFW / GLEW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

// ------------------------------------------
// Simulation grid
static const int NX = 64;
static const int NY = 64;
static const int NZ = 64;

// PDE parameters
static const float DX   = 0.1f;
static const float DT   = 0.0001f;  // smaller dt => more stable
static const float DVAL = 1.0f;
static const float GVAL = 1.0f;     // stronger focusing

// Steps of PDE per frame
static const int STEPS_PER_FRAME = 1;

// ------------------------------------------
// GPU complex math
// ------------------------------------------
__device__ inline float2 cMul(const float2& a, const float2& b)
{
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

// ------------------------------------------
// Nonlinear half-step: ψ <- ψ * exp(i G |ψ|^2 dt/2)
// ------------------------------------------
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

// ------------------------------------------
// Linear step in Fourier space: multiply each mode by exp(-i D k^2 dt)
// ------------------------------------------
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

// ------------------------------------------
// Scale after inverse FFT
// ------------------------------------------
__global__ void scaleKernel(float2* data, float s, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n)
    {
        data[idx].x *= s;
        data[idx].y *= s;
    }
}

// ------------------------------------------
// Compute amplitude
// ------------------------------------------
__global__ void computeAmplitudeKernel(const float2* psi, float* amplitude,
                                       int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < nx*ny*nz)
    {
        float2 val = psi[idx];
        float amp = sqrtf(val.x*val.x + val.y*val.y);
        amplitude[idx] = amp;
    }
}

// ------------------------------------------
// Host function to initialize psi with a swirl
// ------------------------------------------
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
                // swirl around z => phase = atan2f(y - centerY, x - centerX)
                float theta = atan2f(dy_, dx_);
                float re = amp*cosf(theta);
                float im = amp*sinf(theta);
                psiHost[idx] = make_float2(re, im);
            }
        }
    }
}

// ------------------------------------------
// We'll do a bounding box from [-0.5..+0.5]^3
// We'll sample a single slice at z=0.5 in the fragment shader
// But we will SHIFT the volume so the vortex peak is at the domain center
// in local coords. We'll do that by a translation in the model matrix.
//
// Vertex shader: pass fragPos
// Fragment shader: single slice
// ------------------------------------------

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

// We'll just sample z=0.5 in texture coords, ignoring fragPos.z
static const char* fragmentShaderSource = R"(
#version 330 core
in vec3 fragPos;
out vec4 outColor;

uniform sampler3D uVolume;

// The volume is in [-0.5..+0.5]^3 after our model transform
// We sample xCoord, yCoord in [0..1], zCoord=0.5
void main()
{
    float xCoord = fragPos.x + 0.5;
    float yCoord = fragPos.y + 0.5;
    if(xCoord < 0.0 || xCoord > 1.0 ||
       yCoord < 0.0 || yCoord > 1.0)
    {
        outColor = vec4(0,0,0,1);
        return;
    }
    float zCoord = 0.5;

    float amp = texture(uVolume, vec3(xCoord, yCoord, zCoord)).r;
    outColor = vec4(amp, amp, amp, 1.0);
}
)";

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success){
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
    }
    return shader;
}

static GLuint cubeVAO=0, cubeVBO=0;
static void createCube()
{
    float vertices[] = {
        // same 36-vertex cube in [-0.5..+0.5]^3
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,

        -0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,

         0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,

        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    glBindVertexArray(0);
}

static GLuint amplitudeTex=0;
static GLuint prog=0;

static GLuint createShaderProgram()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint sp = glCreateProgram();
    glAttachShader(sp, vs);
    glAttachShader(sp, fs);
    glLinkProgram(sp);

    GLint success;
    glGetProgramiv(sp, GL_LINK_STATUS, &success);
    if(!success){
        char log[1024];
        glGetProgramInfoLog(sp, 1024, nullptr, log);
        std::cerr << "Program link error:\n" << log << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return sp;
}

// We'll upload amplitude to the 3D texture
static void upload3DTexture(const float* data, int nx, int ny, int nz)
{
    glBindTexture(GL_TEXTURE_3D, amplitudeTex);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, nx, ny, nz,
                    GL_RED, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_3D, 0);
}

// Find the maximum amplitude index on the host
// Return (peakX, peakY, peakZ)
static void findPeakAmplitude(const std::vector<float>& ampHost,
                              int nx, int ny, int nz,
                              int& peakX, int& peakY, int& peakZ)
{
    float maxVal=0.0f;
    peakX=0; peakY=0; peakZ=0;
    for(int z=0; z<nz; z++){
        for(int y=0; y<ny; y++){
            for(int x=0; x<nx; x++){
                int idx=(z*ny + y)*nx + x;
                float val=ampHost[idx];
                if(val>maxVal){
                    maxVal=val;
                    peakX=x; peakY=y; peakZ=z;
                }
            }
        }
    }
}

int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    int winW=800, winH=600;
    GLFWwindow* window = glfwCreateWindow(winW, winH, "DVRIPE 3D Follow Vortex", nullptr, nullptr);
    if(!window){
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 2) GLEW
    GLenum glewErr = glewInit();
    if(glewErr != GLEW_OK){
        std::cerr << "GLEW error: " << glewGetErrorString(glewErr) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 3) PDE setup
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);
    initializePsiHost(psiHost, NX, NY, NZ, DX);

    float2* psiDev;
    cudaMalloc(&psiDev, n*sizeof(float2));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(float2), cudaMemcpyHostToDevice);

    // cuFFT
    cufftHandle plan;
    if(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C) != CUFFT_SUCCESS){
        std::cerr << "cufftPlan3d failed!\n";
        return -1;
    }

    float dkx = 2.0f*M_PI/(NX*DX);
    float dky = 2.0f*M_PI/(NY*DX);
    float dkz = 2.0f*M_PI/(NZ*DX);

    // amplitude device array
    float* ampDev;
    cudaMalloc(&ampDev, n*sizeof(float));

    // 4) Create the 3D texture
    glGenTextures(1, &amplitudeTex);
    glBindTexture(GL_TEXTURE_3D, amplitudeTex);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, NX, NY, NZ, 0,
                 GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    // 5) Create bounding box + shader
    createCube();
    prog = createShaderProgram();

    glUseProgram(prog);
    GLint locVol = glGetUniformLocation(prog,"uVolume");
    glUniform1i(locVol, 0);
    glUseProgram(0);

    // PDE kernel config
    int blockSize=256;
    int gridSize=(n+blockSize-1)/blockSize;

    // We'll do a simple orbit camera
    float angleY=0.0f;

    // disable depth so we see the slice
    glDisable(GL_DEPTH_TEST);

    while(!glfwWindowShouldClose(window))
    {
        // PDE steps
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // half-step
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();

            // forward FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_FORWARD);

            // linear
            linearStepKernel<<<gridSize, blockSize>>>(psiDev, NX, NY, NZ, DT, DVAL, dkx, dky, dkz);
            cudaDeviceSynchronize();

            // inverse FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_INVERSE);

            // scale
            float scale = 1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<gridSize, blockSize>>>(psiDev, scale, n);
            cudaDeviceSynchronize();

            // half-step
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();
        }

        // compute amplitude
        computeAmplitudeKernel<<<gridSize, blockSize>>>(psiDev, ampDev, NX, NY, NZ);
        cudaDeviceSynchronize();

        // copy to host
        std::vector<float> ampHost(n);
        cudaMemcpy(ampHost.data(), ampDev, n*sizeof(float), cudaMemcpyDeviceToHost);

        // find peak amplitude index
        int peakX=0, peakY=0, peakZ=0;
        findPeakAmplitude(ampHost, NX, NY, NZ, peakX, peakY, peakZ);

        // convert that index to a shift in [-0.5..+0.5]
        // i in [0..NX-1], so normalizedCoord = (i + 0.5f)/NX
        float fx = (peakX + 0.5f)/(float)NX;
        float fy = (peakY + 0.5f)/(float)NY;
        float fz = (peakZ + 0.5f)/(float)NZ;
        // shift so that (fx, fy, fz) => (0.5, 0.5, 0.5)
        // means we want a translation of (0.5 - fx, etc.)
        float shiftX = 0.5f - fx;
        float shiftY = 0.5f - fy;
        float shiftZ = 0.5f - fz;

        // upload amplitude to 3D texture
        upload3DTexture(ampHost.data(), NX, NY, NZ);

        // revolve camera
        angleY += 0.01f;

        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT);

        // basic perspective
        float fov=60.0f*(M_PI/180.0f);
        float aspect=(float)winW/(float)winH;
        float nearP=0.1f, farP=10.0f;
        auto matIdentity=[](){
            float m[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            return std::vector<float>(m,m+16);
        };
        auto matMultiply=[](const std::vector<float>&A,const std::vector<float>&B){
            std::vector<float> M(16,0);
            for(int r=0;r<4;r++){
                for(int c=0;c<4;c++){
                    for(int k=0;k<4;k++){
                        M[r*4+c]+=A[r*4+k]*B[k*4+c];
                    }
                }
            }
            return M;
        };

        float f = 1.0f/tanf(fov*0.5f);
        std::vector<float> matP(16,0);
        matP[0]=f/aspect; matP[5]=f;
        matP[10]=(farP+nearP)/(nearP-farP);
        matP[11]=-1;
        matP[14]=(2.0f*farP*nearP)/(nearP-farP);

        // minimal orbit camera
        auto sub3=[](float x1,float y1,float z1,float x2,float y2,float z2){
            return std::vector<float>{x1-x2,y1-y2,z1-z2};
        };
        auto norm3=[](const std::vector<float>& v){
            float l=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
            return std::vector<float>{v[0]/l,v[1]/l,v[2]/l};
        };
        auto cross3=[](const std::vector<float>&a,const std::vector<float>&b){
            return std::vector<float>{
                a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]
            };
        };

        float dist=1.5f;
        float eyex = dist*sinf(angleY);
        float eyey = 0.0f;
        float eyez = dist*cosf(angleY);
        std::vector<float> eye={eyex,eyey,eyez};
        std::vector<float> center={0,0,0};
        std::vector<float> up={0,1,0};

        auto fwd=norm3(sub3(center[0],center[1],center[2], eye[0],eye[1],eye[2]));
        auto rht=norm3(cross3(fwd, up));
        auto u2 = cross3(rht, fwd);

        std::vector<float> matV=matIdentity();
        matV[0]=rht[0]; matV[1]=u2[0]; matV[2]=-fwd[0];
        matV[4]=rht[1]; matV[5]=u2[1]; matV[6]=-fwd[1];
        matV[8]=rht[2]; matV[9]=u2[2]; matV[10]=-fwd[2];

        float tx=-(rht[0]*eye[0]+rht[1]*eye[1]+rht[2]*eye[2]);
        float ty=-(u2[0]*eye[0]+u2[1]*eye[1]+u2[2]*eye[2]);
        float tz= (fwd[0]*eye[0]+fwd[1]*eye[1]+fwd[2]*eye[2]);
        matV[12]=tx; matV[13]=ty; matV[14]=tz;

        // model transform => shift volume by (shiftX, shiftY, shiftZ)
        // so that the vortex peak is at (0.5, 0.5, 0.5) in local coords
        // => [-0.5..+0.5]^3 => means we do a translation of shift?
        // We define a minimal translation matrix:
        auto matTranslate=[&](float sx,float sy,float sz){
            std::vector<float> T=matIdentity();
            T[12]=sx; T[13]=sy; T[14]=sz;
            return T;
        };
        auto matM=matTranslate(shiftX, shiftY, shiftZ);

        // combine
        auto matVM=matMultiply(matV, matM);
        auto matPVM=matMultiply(matP, matVM);

        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        GLint locMVP= glGetUniformLocation(prog,"uMVP");
        glUniformMatrix4fv(locMVP,1,GL_FALSE, matPVM.data());

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, amplitudeTex);

        glBindVertexArray(cubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cufftDestroy(plan);
    cudaFree(psiDev);
    cudaFree(ampDev);

    glDeleteTextures(1, &amplitudeTex);
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteBuffers(1, &cubeVBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
