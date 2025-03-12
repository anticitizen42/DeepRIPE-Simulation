// DVRIPE Simulation - CUDA + OpenGL Visualization (GLFW + GLEW)

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

#define WIDTH 512
#define HEIGHT 512
#define DT 0.05f       // increased for visible change
#define DIFF 0.25f     // increased for visible diffusion

GLuint pbo;
cudaGraphicsResource* cuda_pbo_resource;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__global__ void dvripePDEKernel(float* field, float* field_next, float dt, float diff, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int xp1 = (x + 1) % width;
    int xm1 = (x - 1 + width) % width;
    int yp1 = (y + 1) % height;
    int ym1 = (y - 1 + height) % height;

    float laplace = field[yp1 * width + x] + field[ym1 * width + x] +
                    field[y * width + xp1] + field[y * width + xm1] - 4.0f * field[idx];

    field_next[idx] = field[idx] + diff * laplace * dt;
}

__global__ void visualizeKernel(float* field, uchar4* pixels, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float val = field[idx];
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    unsigned char intensity = (unsigned char)(val * 255.0f);

    pixels[idx] = make_uchar4(intensity, intensity, intensity, 255);
}

void initGL(){
    if (!glfwInit()){
        std::cerr << "Failed to initialize GLFW\n";
        exit(EXIT_FAILURE);
    }
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "DVRIPE Simulation", nullptr, nullptr);
    if (!window){
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK){
        std::cerr << "Failed to initialize GLEW\n";
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
}

void initPBO(){
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

int main(){
    initGL();
    initPBO();

    float *d_field, *d_field_next;
    gpuErrchk(cudaMalloc(&d_field, WIDTH * HEIGHT * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_field_next, WIDTH * HEIGHT * sizeof(float)));

    float host_field[WIDTH * HEIGHT] = {0};
    int cx = WIDTH / 2, cy = HEIGHT / 2;
    float radius = 50.0f;
    for (int y=0; y<HEIGHT; y++){
        for (int x=0; x<WIDTH; x++){
            float dist = sqrtf((x-cx)*(x-cx) + (y-cy)*(y-cy));
            if (dist < radius)
                host_field[y*WIDTH+x] = 1.0f - (dist / radius);
        }
    }

    gpuErrchk(cudaMemcpy(d_field, host_field, WIDTH*HEIGHT*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_field_next, d_field, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToDevice));

    GLFWwindow* window = glfwGetCurrentContext();
    dim3 blockSize(16,16);
    dim3 gridSize((WIDTH+15)/16,(HEIGHT+15)/16);

    while (!glfwWindowShouldClose(window)){
        for(int i=0;i<10;i++){
            dvripePDEKernel<<<gridSize,blockSize>>>(d_field,d_field_next,DT,DIFF,WIDTH,HEIGHT);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
            std::swap(d_field,d_field_next);
        }

        uchar4* d_pixels;
        size_t size;
        gpuErrchk(cudaGraphicsMapResources(1,&cuda_pbo_resource,0));
        gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_pixels,&size,cuda_pbo_resource));

        visualizeKernel<<<gridSize,blockSize>>>(d_field,d_pixels,WIDTH,HEIGHT);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaGraphicsUnmapResources(1,&cuda_pbo_resource,0));

        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(WIDTH,HEIGHT,GL_RGBA,GL_UNSIGNED_BYTE,0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    gpuErrchk(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glDeleteBuffers(1,&pbo);
    gpuErrchk(cudaFree(d_field));
    gpuErrchk(cudaFree(d_field_next));
    glfwTerminate();
    return 0;
}
