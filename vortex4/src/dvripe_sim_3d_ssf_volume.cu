/******************************************************************************
 * dvripe_sim_3d_ssf_volume.cu
 *
 * Demonstrates:
 *   1) 3D Split-Step Fourier integrator for the focusing Schrödinger PDE:
 *         i ∂ψ/∂t = -D ∇²ψ - G |ψ|² ψ
 *      with periodic boundaries.
 *   2) Real-time volume rendering of the amplitude in a 3D texture
 *      using a minimal raymarching fragment shader in OpenGL.
 *
 * Build on Windows (example):
 *   nvcc dvripe_sim_3d_ssf_volume.cu -o dvripe_sim_3d_ssf_volume.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build on Linux (example):
 *   nvcc dvripe_sim_3d_ssf_volume.cu -o dvripe_sim_3d_ssf_volume \
 *       -lcufft -lGLEW -lglfw -lGL
 *
 * Caveats:
 *   - This is a large, advanced example.
 *   - 3D SSF can be expensive; volume raymarching is also expensive.
 *   - Keep grid small (64³ or less) for any hope of real-time.
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

// ---------- OpenGL / GLFW / GLEW ----------
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

// For string storing shaders
#include <string>

// ------------------------------------------
// Simulation parameters
// ------------------------------------------
static const int NX = 64;
static const int NY = 64;
static const int NZ = 64;

static const float DX = 0.1f;
static const float DT = 0.0001f;

static const float DVAL = 1.0f;
static const float GVAL = 0.1f;

// Steps of PDE per frame
static const int STEPS_PER_FRAME = 0;

// We'll do periodic boundaries

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
// Nonlinear half-step: ψ <- ψ * exp(i * G * |ψ|^2 * dt/2)
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
// Scale kernel (for inverse FFT normalization)
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
// We'll create a float3D amplitude array after PDE each frame
// Then we upload to a 3D texture
// amplitude[x,y,z] = sqrt( real^2 + imag^2 )
// ------------------------------------------
__global__ void computeAmplitudeKernel(const float2* psi,
                                       float* amplitude,
                                       int nx, int ny, int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < (nx*ny*nz))
    {
        float2 val = psi[idx];
        float amp = sqrtf(val.x*val.x + val.y*val.y);
        amplitude[idx] = amp;
    }
}

// ------------------------------------------
// Host function to initialize psi
// We'll do a wide Gaussian + swirl
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

                float amp = expf(-r2/(nx*dx*0.3f));
                // swirl around z
                float theta = atan2f(dy_, dx_);
                float re = amp*cosf(theta);
                float im = amp*sinf(theta);
                psiHost[idx] = make_float2(re, im);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// We'll do a minimal volume raymarch with a fragment shader that reads from a
// 3D texture containing amplitude. The camera orbits around the volume.
//
// 1) We'll store amplitude in a 3D texture amplitudeTex.
// 2) The fragment shader does a front-to-back raymarch from -Z to +Z in local
//    coords, sampling amplitude. We'll color map amplitude -> simple RGBA.
//
// This is not optimized. It's just a demonstration. For bigger grids or
// more advanced shading, you'll need a better approach.
// -----------------------------------------------------------------------------

// We'll store the GLSL shader code in strings
static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 inPos;
out vec3 fragPos;
uniform mat4 uMVP;
void main()
{
    fragPos = inPos; // pass to fragment
    gl_Position = uMVP * vec4(inPos, 1.0);
}
)";

static const char* fragmentShaderSource = R"(
#version 330 core
in vec3 fragPos;
out vec4 outColor;

// The 3D amplitude texture
uniform sampler3D uVolume;
// volume dimensions in texture coords are [0..1]
uniform float uStepSize;   // step for raymarch
uniform vec3 uEyePos;      // camera position in object coords
uniform mat4 uInvModel;    // for direction transform

// a basic front->back raymarch
void main()
{
    // We'll do a naive raymarch from -1..+1 in z, or we can do a bounding box approach.
    // Let's find the ray direction in object space
    vec3 dir = normalize(fragPos - uEyePos);

    // We'll do a fixed number of steps, e.g. 200
    const int MAX_STEPS = 20;
    float t = 0.0;
    vec4 accumColor = vec4(0.0);

    vec3 startPos = uEyePos;
    // We'll step forward in small increments
    for(int i=0; i<MAX_STEPS; i++)
    {
        vec3 samplePos = startPos + dir * t;
        // transform samplePos from [-NX/2..NX/2] etc. into [0..1]
        // We'll assume the volume is in [-0.5..+0.5] in each axis for simplicity
        // If we place a bounding box of side 1
        vec3 texCoord = samplePos + vec3(0.5, 0.5, 0.5);

        // If outside [0..1], break
        if(any(lessThan(texCoord, vec3(0.0))) ||
           any(greaterThan(texCoord, vec3(1.0))))
        {
            break;
        }

        // sample amplitude
        float amp = texture(uVolume, texCoord).r;
        // color map: let's do grayscale
        // we can do alpha = 1 - exp(-amp * factor)
        float alpha = clamp(amp*0.05, 0.0, 1.0);
        vec3 color = vec3(amp); // grayscale

        // front->back compositing
        accumColor.rgb += (1.0 - accumColor.a)*color*alpha;
        accumColor.a += (1.0 - accumColor.a)*alpha;

        if(accumColor.a > 0.95)
            break; // near opaque

        t += uStepSize;
    }

    outColor = accumColor;
}
)";

// We'll compile these shaders into a program
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    // check errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success){
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << std::endl;
    }
    return shader;
}

// We'll make a simple cube VAO
static GLuint cubeVAO=0, cubeVBO=0;
static void createCube()
{
    float vertices[] = {
        // a simple cube from [-0.5..0.5] in each axis
        // We'll just draw a big quad that encloses the entire volume for raymarch
        // Actually, let's do a single full-screen quad approach, but let's do a cube approach for demonstration

        // 8 corners, but let's do 36 for a typical cube approach
        // We'll let the fragment shader do the raymarch inside
        // Actually, let's do a single 2-triangle approach if we only want a bounding box
        // We'll do a small cube so we can see the bounding box from outside
        // For simplicity, let's just define 36 vertices for a cube

        // position
        // front face
        -0.5f, -0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

         0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,
        -0.5f, -0.5f,  0.5f,

        // back face
        -0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,

         0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,

        // left face
        -0.5f,  0.5f,  0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f, -0.5f, -0.5f,

        -0.5f, -0.5f, -0.5f,
        -0.5f, -0.5f,  0.5f,
        -0.5f,  0.5f,  0.5f,

        // right face
         0.5f,  0.5f,  0.5f,
         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f, -0.5f,

         0.5f, -0.5f, -0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f, -0.5f,  0.5f,

        // top face
        -0.5f,  0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,
         0.5f,  0.5f, -0.5f,

         0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f, -0.5f,
        -0.5f,  0.5f,  0.5f,

        // bottom face
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

    // position at location=0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);

    glBindVertexArray(0);
}

// We'll create a 3D texture to store amplitude
static GLuint amplitudeTex=0;

// We'll create a shader program
static GLuint prog=0;

static GLuint createShaderProgram()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint sp = glCreateProgram();
    glAttachShader(sp, vs);
    glAttachShader(sp, fs);
    glLinkProgram(sp);

    // check link
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

// We'll do a minimal orbit camera
// revolve around Y axis
static float angleY=0.0f;

static void upload3DTexture(const float* data, int nx, int ny, int nz)
{
    glBindTexture(GL_TEXTURE_3D, amplitudeTex);
    glTexSubImage3D(GL_TEXTURE_3D, 0,
                    0, 0, 0, nx, ny, nz,
                    GL_RED, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_3D, 0);
}

int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    // create a bigger window so we can see the volume
    int winW=800, winH=600;
    GLFWwindow* window = glfwCreateWindow(winW, winH, "DVRIPE 3D SSF + Volume Raymarch", nullptr, nullptr);
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

    // 3) PDE setup
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);
    initializePsiHost(psiHost, NX, NY, NZ, DX);

    float2* psiDev;
    cudaMalloc(&psiDev, n*sizeof(float2));
    cudaMemcpy(psiDev, psiHost.data(), n*sizeof(float2), cudaMemcpyHostToDevice);

    // cuFFT plan
    cufftHandle plan;
    if(cufftPlan3d(&plan, NZ, NY, NX, CUFFT_C2C) != CUFFT_SUCCESS){
        std::cerr << "cufftPlan3d failed!\n";
        return -1;
    }

    // wave numbers
    float dkx = 2.0f*M_PI/(NX*DX);
    float dky = 2.0f*M_PI/(NY*DX);
    float dkz = 2.0f*M_PI/(NZ*DX);

    // 4) We'll allocate a float array for amplitude
    float* ampDev;
    cudaMalloc(&ampDev, n*sizeof(float));

    // 5) Create the 3D texture for amplitude
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

    // 6) Create VAO for cube
    createCube();

    // 7) Create shader program
    prog = createShaderProgram();

    // 8) Setup some uniforms
    glUseProgram(prog);
    GLint locVol = glGetUniformLocation(prog, "uVolume");
    glUniform1i(locVol, 0); // texture unit 0
    glUseProgram(0);

    // PDE kernel config
    int blockSize=256;
    int gridSize=(n+blockSize-1)/blockSize;

    // main loop
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

            // linear step
            linearStepKernel<<<gridSize, blockSize>>>(psiDev, NX, NY, NZ, DT, DVAL, dkx, dky, dkz);
            cudaDeviceSynchronize();

            // inverse FFT
            cufftExecC2C(plan, (cufftComplex*)psiDev, (cufftComplex*)psiDev, CUFFT_INVERSE);

            // scale
            float scale=1.0f/(float)(NX*NY*NZ);
            scaleKernel<<<gridSize, blockSize>>>(psiDev, scale, n);
            cudaDeviceSynchronize();

            // half-step
            nonlinearHalfStepKernel<<<gridSize, blockSize>>>(psiDev, n, DT, GVAL);
            cudaDeviceSynchronize();
        }

        // compute amplitude
        computeAmplitudeKernel<<<gridSize, blockSize>>>(psiDev, ampDev, NX, NY, NZ);
        cudaDeviceSynchronize();

        // copy amplitude to host
        // or directly map it to a 3D texture with cudaGraphicsResource?
        // For simplicity, we'll do a host copy. For better performance, do PBO approach.
        std::vector<float> ampHost(n);
        cudaMemcpy(ampHost.data(), ampDev, n*sizeof(float), cudaMemcpyDeviceToHost);

        // upload to 3D texture
        upload3DTexture(ampHost.data(), NX, NY, NZ);

        // camera orbit
        angleY += 0.01f;
        // We'll define a simple model matrix that places volume in [-0.5..0.5]^3
        // Then a view+proj that orbits around Y

        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(prog);

        // compute MVP
        float dist=1.5f; // distance from center
        float eyex = dist*sinf(angleY);
        float eyey = 0.0f;
        float eyez = dist*cosf(angleY);

        // basic perspective
        float fov=60.0f*(M_PI/180.0f);
        float aspect = (float)winW/(float)winH;
        float nearP=0.1f, farP=10.0f;
        // build a simple perspective * lookAt * model matrix in CPU for clarity
        auto matIdentity=[](){
            float m[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
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
        // perspective
        float f = 1.0f/tanf(fov*0.5f);
        std::vector<float> matP(16,0);
        matP[0]=f/aspect; matP[5]=f; matP[10]=(farP+nearP)/(nearP-farP);
        matP[11]=-1; matP[14]=(2.0f*farP*nearP)/(nearP-farP);

        // lookAt from (ex,ey,ez) to (0,0,0), up(0,1,0)
        // we'll do a minimal approach
        float3 eye=make_float3(eyex,eyey,eyez);
        float3 center=make_float3(0,0,0);
        float3 up=make_float3(0,1,0);
        auto sub3=[](float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);};
        auto norm3=[](float3 a){float l=sqrtf(a.x*a.x+a.y*a.y+a.z*a.z);return make_float3(a.x/l,a.y/l,a.z/l);};
        auto cross3=[](float3 a,float3 b){
            return make_float3(a.y*b.z - a.z*b.y,
                               a.z*b.x - a.x*b.z,
                               a.x*b.y - a.y*b.x);
        };
        float3 fwd=norm3(sub3(center,eye));
        float3 rht=norm3(cross3(fwd,up));
        float3 u2 = cross3(rht,fwd);
        std::vector<float> matV=matIdentity();
        matV[0]=rht.x; matV[1]=u2.x; matV[2]=-fwd.x;
        matV[4]=rht.y; matV[5]=u2.y; matV[6]=-fwd.y;
        matV[8]=rht.z; matV[9]=u2.z; matV[10]=-fwd.z;
        // translate
        matV[12]=- (rht.x*eye.x + rht.y*eye.y + rht.z*eye.z);
        matV[13]=- (u2.x*eye.x + u2.y*eye.y + u2.z*eye.z);
        matV[14]=  (fwd.x*eye.x + fwd.y*eye.y + fwd.z*eye.z);

        // model: identity that puts volume in [-0.5..0.5]
        auto matM=matIdentity();

        auto matVM = matMultiply(matV, matM);
        auto matPVM= matMultiply(matP, matVM);

        // upload to uniform
        GLint locMVP= glGetUniformLocation(prog,"uMVP");
        glUniformMatrix4fv(locMVP,1,GL_FALSE, matPVM.data());

        // set sampler
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, amplitudeTex);

        // step size for raymarch
        float stepSize=1.0f/64.0f;
        GLint locStep= glGetUniformLocation(prog,"uStepSize");
        glUniform1f(locStep, stepSize);

        // Eye pos in object coords => eye is in [-0.5..0.5]? We'll just pass (ex,ey,ez) plus 0.5
        // Actually, we placed the volume in [-0.5..0.5], so if eye is outside that range,
        // we can pass it as is
        GLint locEye= glGetUniformLocation(prog,"uEyePos");
        glUniform3f(locEye, eyex, eyey, eyez);

        // draw the cube
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
