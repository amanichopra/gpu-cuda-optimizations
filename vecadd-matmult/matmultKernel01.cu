///
/// matmultKernel01.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

#define FOOTPRINT_SIZE BLOCK_SIZE

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    int matrixSize = A.width;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float result = 0;

    // Loop over the tiles of the input matrices
    for (int i = 0; i < matrixSize / BLOCK_SIZE; ++i) {
        // Load tiles of matrices A and B into shared memory
        sA[ty][tx] = A.elements[row * matrixSize + i * BLOCK_SIZE + tx];
        sB[ty][tx] = B.elements[(i * BLOCK_SIZE + ty) * matrixSize + col];

        __syncthreads();

        // Perform computation within the tile
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            result += sA[ty][j] * sB[j][tx];
        }

        __syncthreads();
    }

    // Write the computed value to matrix C
    C.elements[row * matrixSize + col] = result;
}

