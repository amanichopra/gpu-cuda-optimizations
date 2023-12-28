#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

__global__ void addArraysKernel(float *A, float *B, float *C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void addArraysOnGPU(int K, int threadsPerBlock, int blocks) {
    int size = K * 1000000;
    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

    // Initialize arrays A and B (for simplicity, just assigning some values)
    for (int i = 0; i < size; ++i) {
        h_A[i] = i * 1.5;
        h_B[i] = i * 2.5;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    switch (blocks) {
        case 1:
            addArraysKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, size);
            break;
        default:
            addArraysKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, size);
            break;
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken for K = " << K << ", Blocks = " << blocks << ", Threads per Block = " << threadsPerBlock << ": " << duration.count() << " seconds" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main(int argc, char **argv) {
    int all_configurations[3][2] = {{1, 1}, {1, 256}, {0, 256}}; // {blocks, threadsPerBlock}
    
    int configurations[2] = {0};
    if (strcmp("scenario-1", argv[1]) == 0) {
        configurations[0] = 1;
        configurations[1] = 1;
    } else if (strcmp("scenario-2", argv[1]) == 0) {
        configurations[0] = 1;
        configurations[1] = 256;
    } else {
        configurations[0] = 0;
        configurations[1] = 256;
    }

    for (int k : {1, 5, 10, 50, 100}) {
        int blocks = configurations[0];
        int threadsPerBlock = configurations[1];

        if (blocks == 0) { // case3 
            int size = k * 1000000;
            blocks = (size + 255) / 256;
        } 

        addArraysOnGPU(k, threadsPerBlock, blocks);
    }

    return 0;
}
