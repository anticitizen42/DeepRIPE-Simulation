/******************************************************************************
 * dvripe_sim_3d_ssf_debug.cu
 *
 * A debug version of the 3D Split-Step Fourier + volume raymarch code.
 *
 * Changes:
 *   1) PDE steps are commented out so the wavefunction is never updated.
 *   2) The fragment shader is replaced with a constant soft color
 *      (e.g., light blue) instead of raymarching the volume.
 *
 * This helps confirm whether geometry is actually being drawn.
 *
 * Build (Windows example):
 *   nvcc dvripe_sim_3d_ssf_debug.cu -o dvripe_sim_3d_ssf_debug.exe ^
 *       -I"C:\vcpkg\installed\x64-windows\include" ^
 *       -L"C:\vcpkg\installed\x64-windows\lib" ^
 *       -lcufft -lglew32 -lglfw3dll -lopengl32
 *
 * Build (Linux example):
 *   nvcc dvripe_sim_3d_ssf_debug.cu -o dvripe_sim_3d_ssf_debug \
 *       -lcufft -lGLEW -lglfw -lGL
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

// We'll keep the same grid size, but PDE won't run anyway
static const int NX = 64;
static const int NY = 64;
static const int NZ = 64;

// We won't use these PDE parameters now, since PDE steps are disabled
static const float DX = 0.1f;
static const float DT = 0.001f;
static const float DVAL = 1.0f;
static const float GVAL = 1.0f;
static const int STEPS_PER_FRAME = 1;

// ------------------------------------------
// Minimal swirl initialization
// We'll just use the built-in float2 from CUDA
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
                float theta = atan2f(dy_, dx_);
                float re = amp*cosf(theta);
                float im = amp*sinf(theta);
                psiHost[idx] = make_float2(re, im);
            }
        }
    }
}

// ------------------------------------------
// We'll keep the bounding-cube approach.
// The fragment shader will NOT do volume raymarching;
// it will just output a soft color.
// ------------------------------------------

// Minimal vertex shader (same as before)
static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 inPos;
uniform mat4 uMVP;
void main()
{
    gl_Position = uMVP * vec4(inPos, 1.0);
}
)";

// Fragment shader: constant soft color (light blue)
static const char* fragmentShaderSource = R"(
#version 330 core
out vec4 outColor;

void main()
{
    // Just output a soft color
    outColor = vec4(0.4, 0.6, 0.8, 1.0);  // a light blue
}
)";

// We'll compile these
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
        // Same 36-vertex cube from [-0.5..0.5]
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
         0.5f,  0.5f, -0.5f,
         0.5f, -0.5f, -0.5f,

         0.5f, -0.5f, -0.5f,
         0.5f, -0.5f,  0.5f,
         0.5f,  0.5f,  0.5f,

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

int main()
{
    // 1) Init GLFW
    if(!glfwInit()){
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }
    // bigger window
    int winW=800, winH=600;
    GLFWwindow* window = glfwCreateWindow(winW, winH, "DVRIPE 3D Debug (Soft Color)", nullptr, nullptr);
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

    // 3) We won't do PDE steps, but let's allocate and init psi anyway
    int n = NX*NY*NZ;
    std::vector<float2> psiHost(n);
    initializePsiHost(psiHost, NX, NY, NZ, DX);

    // (We won't even allocate device memory for psi, or do cuFFT)

    // 4) Create the cube geometry
    createCube();

    // 5) Create the minimal shader program
    prog = createShaderProgram();

    glEnable(GL_DEPTH_TEST);

    // We'll do a simple orbit camera
    float angleY=0.0f;

    while(!glfwWindowShouldClose(window))
    {
        // PDE steps are commented out:
        /*
        for(int step=0; step<STEPS_PER_FRAME; step++)
        {
            // NO PDE updates
        }
        */

        // We'll revolve the camera around Y
        angleY += 0.01f;

        glClearColor(0,0,0,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Basic perspective
        float fov = 60.0f*(M_PI/180.0f);
        float aspect = (float)winW/(float)winH;
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

        // lookAt
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

        auto matM=matIdentity();
        auto matVM=matMultiply(matV, matM);
        auto matPVM=matMultiply(matP, matVM);

        // Use the program
        glUseProgram(prog);

        GLint locMVP= glGetUniformLocation(prog,"uMVP");
        glUniformMatrix4fv(locMVP,1,GL_FALSE, matPVM.data());

        glBindVertexArray(cubeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
