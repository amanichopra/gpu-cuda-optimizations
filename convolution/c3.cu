#include <time.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cudnn.h>
#include <string>

#define checkcuDNN(expression)                               \
{                                                            \
	cudnnStatus_t status = (expression);                     \
	if (status != CUDNN_STATUS_SUCCESS)                      \
	{														 \
		std::cerr << "Error on line " << __LINE__ << ": "    \
				<< cudnnGetErrorString(status) << std::endl; \
		exit(EXIT_FAILURE);                               	 \
	}                                                        \
}

int H = 1024;
int W = 1024;
int C = 3;

int FH = 3; 
int FW = 3;
int K = 64;

int main() {
	double* h_inp = (double*) malloc(H * W * C * sizeof(double));
	double* h_filt = (double*) malloc(K * C * FH * FW * sizeof(double));
	double* h_out = (double*) malloc(H * W * K * sizeof(double));

    double *d_inp = NULL; 
    double *d_filt = NULL;
    double *d_out = NULL;
	
	cudaMalloc(&d_inp, H * W * C * sizeof(double));
	cudaMalloc(&d_filt, K * C * FH * FW * sizeof(double));
	cudaMalloc(&d_out, H * W * K * sizeof(double));


    // initialize input
    for (int c = 0; c < C; c++) {
		for (int h = 0; h < H; h++) {
			for(int w = 0; w < W; w++) {
				h_inp[(W * H * c) +(h * W) + w]= c * (w + h); // c*(x+y) as mentioned in instructions
			}
		}
	}

    // initialize kernel
    for(int k = 0; k < K; k++) {
		for(int c = 0; c < C; c++) {
			for(int h = 0; h < FH; h++) {
				for(int w=0; w < FW; w++) {
					h_filt[(k * C * FW * FH) + (FW * FH * c) + (h * FW) + w] = (c + k) * (w + h); // (c+k)*(i*j) as mentioned in instructions
 				}
			}
		}
	}

	cudaMemcpy(d_inp, h_inp, H * W * C * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudaMemcpy(d_filt, h_filt, K * C * FH * FW * sizeof(double), cudaMemcpyHostToDevice);

	cudnnHandle_t cudnn;
	checkcuDNN(cudnnCreate(&cudnn));

	cudnnTensorDescriptor_t input_descriptor;
	checkcuDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkcuDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

	cudnnFilterDescriptor_t filter_descriptor;
	checkcuDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
	cudnnSetFilter4dDescriptor(filter_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

	cudnnTensorDescriptor_t output_descriptor;
	checkcuDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkcuDNN(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkcuDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkcuDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, FH / 2, FW / 2, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

	
    //cudnnConvolutionFwdAlgo_t convAlg = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    int requestedAlgCount = 8; // 8 algs here: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionFwdAlgo_t
    int returnedAlgoCount = -1;
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm_results[2 * requestedAlgCount];

	checkcuDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, requestedAlgCount, &returnedAlgoCount, convolution_algorithm_results));
    
    cudnnConvolutionFwdAlgo_t convAlg = convolution_algorithm_results[0].algo;
    
    using namespace std;
    
    string convAlgName;
    switch ((int)convAlg) {
        case 0:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
            break;
        case 2:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
            break;
        case 3:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
            break;
        case 4:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
            break;
        case 5:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
            break;
        case 6:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
            break;
        case 7:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
            break;
        case 8:
            convAlgName = "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
            break;
    }

    printf("Using convolution algorithm: %s (described at https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionFwdAlgo_t)\n", convAlgName.c_str());

	size_t workspace_size = 0;
	checkcuDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_descriptor, filter_descriptor, convolution_descriptor, output_descriptor, convAlg, &workspace_size));

	void  *workspace;
	cudaMalloc(&workspace, workspace_size);

	double alpha = 1.0;
    double beta = 0.0;
    struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	checkcuDNN(cudnnConvolutionForward(cudnn, &alpha, input_descriptor, d_inp, filter_descriptor, d_filt, convolution_descriptor, convAlg, workspace, workspace_size, &beta, output_descriptor, d_out));

	clock_gettime(CLOCK_MONOTONIC, &end);

	double time = (end.tv_sec - start.tv_sec) +(end.tv_nsec - start.tv_nsec) / 1E9;
    printf("cuDNN Convolution Time: %lf seconds\n", time);

	cudaMemcpy(h_out, d_out, H * W * K * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

    double checksum = 0.0;
    for (int k = 0; k < K; k++) {
        for(int i = 0; i < H; i++) {
            for(int j = 0; j < W; j++) {
                checksum += h_out[(H * W * k) + (i * W) + j];
            }
        }
    }

	printf("cuDNN Convolution Output Checksum: %lf\n", checksum);

	free(h_inp);
	free(h_filt);
	free(h_out);
	cudaFree(d_inp);
	cudaFree(d_filt);
	cudaFree(d_out);
    cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(filter_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
	cudnnDestroy(cudnn);
	
	return 0;
}
