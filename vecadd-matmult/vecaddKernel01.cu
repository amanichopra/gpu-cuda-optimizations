///
/// vecAddKernel01.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// By David Newman
/// Created: 2011-02-16
/// Last Modified: 2011-02-16 DVN
///
/// This Kernel adds two Vectors A and B in C on GPU
//  using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStartIndex  = blockIdx.x * blockDim.x * N;
    int stride = blockDim.x;
    int threadStartIndex = blockStartIndex + threadIdx.x;
    for(int i=0; i<N; ++i ){
        C[threadStartIndex + stride*i] = A[threadStartIndex + stride*i] + B[threadStartIndex + stride*i];
    }
}
