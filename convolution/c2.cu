#include <time.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int H = 1024;
int W = 1024;
int C = 3;

int FH = 3; 
int FW = 3;
int K = 64;

int BLOCK_SIZE = 4;

__global__ void tiled_convolution(double *inp, int H, int W, int C, double *filter, int FH, int FW, int K, double *out, int block_size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // set up tiling and shared memory
	int tile_width = block_size + (FW / 2) * 2;
	int tile_height = block_size + (FH / 2) * 2;
	extern __shared__ double tile[];

    // boundary check
	if (i < H && j < W && k < K) {
        int tile_i = threadIdx.x + FH / 2;
		int tile_j = threadIdx.y + FW / 2;
		
		for (int c = 0; c < C; ++c) {
			
			tile[tile_j + (tile_i * tile_width) + (c * tile_height * tile_width)] = inp[j + (i * W) + (c * H * W)];

            // EDGE CASES
			// top of input
			if (threadIdx.x == 0) {
				if (i > 0) {
					tile[tile_j + (tile_i - 1) * tile_width + c * tile_height * tile_width] = inp[j + (i - 1) * W + c * H * W];
				}
				else {
					tile[tile_j + (tile_i - 1) * tile_width + c * tile_height * tile_width] = 0.0;
				}

				// left corner
				if (threadIdx.y == 0) {
					if (j > 0 && i > 0) {
						tile[(tile_j - 1) + (tile_i - 1) * tile_width + c * tile_height * tile_width] = inp[(j - 1) + (i - 1) * W + c * H * W];
					}
					else {
						tile[(tile_j - 1) + (tile_i - 1) * tile_width + c * tile_height * tile_width] = 0.0;
					}

				}
			}

			// right side of input
			if (threadIdx.y == block_size - 1) {
				if (j > W - 1) {
                    tile[(tile_j + 1) + tile_i * tile_width + c * tile_height * tile_width] = 0.0;
					tile[(tile_j + 1) + tile_i * tile_width + c * tile_height * tile_width] = inp[(j + 1) + i * W + c * H * W];
				}

				else {
					tile[(tile_j + 1) + tile_i * tile_width + c * tile_height * tile_width] = inp[(j + 1) + i * W + c * H * W];
				}

				// right corner
				if (threadIdx.x == 0) {
					if (j < W - 1 && i > 0) {
						tile[(tile_j + 1) + (tile_i - 1) * tile_width + c * tile_height * tile_width] = inp[(j + 1) + (i - 1) * W + c * H * W];
					}
					else {
						tile[(tile_j + 1) + (tile_i - 1) * tile_width + c * tile_height * tile_width] = 0.0;
					}
				}
			}

			// bottom of inp
			if (threadIdx.x == block_size - 1) {
				if (i < H - 1) {
					tile[tile_j + (tile_i + 1) * tile_width + c * tile_height * tile_width] =   inp[j + (i + 1) * W + c * H * W];
				}
				else {
					tile[tile_j + (tile_i + 1) * tile_width + c * tile_height * tile_width] = 0.0;
				}

				// right corner
				if (threadIdx.y == block_size - 1) {
					if (j < W - 1 && i < H - 1)
					{
						tile[(tile_j + 1) + (tile_i + 1) * tile_width + c * tile_height * tile_width] =  inp[(j + 1) + (i + 1) * W + c * H * W];
					}
					else {
						tile[(tile_j + 1) + (tile_i + 1) * tile_width + c * tile_height * tile_width] = 0.0;

					}
				}
			}

			// left side of inp
			if (threadIdx.y == 0) {
				if (j > 0) {
					tile[(tile_j - 1) + tile_i * tile_width + c * tile_height * tile_width] =  inp[(j - 1) + i * W + c * H * W];
				}
				else {
					tile[(tile_j - 1) + tile_i * tile_width + c * tile_height * tile_width] = 0.0;
				}

				// bottom corner
				if (threadIdx.x == block_size - 1) {
					if (j > 0 && i < H - 1) {
						tile[(tile_j - 1) + (tile_i + 1) * tile_width + c * tile_height * tile_width] = inp[(j - 1) + (i + 1) * W + c * H * W];
					}
					else {
						tile[(tile_j - 1) + (tile_i + 1) * tile_width + c * tile_height * tile_width] = 0.0;
					}
				}
			}
		}

		__syncthreads();

		int out_idx = j + i * W + k * H * W;
		
		int row = i - (FH / 2), col = j - (FW / 2);
		double conv_val = 0.0;

		for (int c = 0; c < C; ++c) {
			for (int fh = 0; fh < FH; ++fh) 
			{
				for (int fw = 0; fw < FW; ++fw) 
				{

					if (col + fw >= 0 && col + fw < W && row + fh >= 0 && row + fh < H) 
					{
						int in_idx = (col + fw) + (row + fh) * W + c * H * W;

						int f_idx = (FW - 1 - fw) + (FH - 1 - fh) * FW + c * FH * FW + k * C * FH * FW;
						conv_val += inp[in_idx] * filter[f_idx];
					}
					
				}
			}
		}

		out[out_idx] = conv_val;
	}
}

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

	dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE, K);
	dim3 num_blocks(H / BLOCK_SIZE, W / BLOCK_SIZE);

    // shared mmemory size
	int shared_mem_size = (BLOCK_SIZE + (FH / 2) * 2) * (BLOCK_SIZE + (FW / 2) * 2) * C * sizeof(double);

    struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
	tiled_convolution<<<num_blocks, num_threads, shared_mem_size>>>(d_inp, H, W, C, d_filt, FH, FW, K, d_out, BLOCK_SIZE);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	double time = (end.tv_sec - start.tv_sec) +(end.tv_nsec - start.tv_nsec) / 1E9;
    printf("Tiled Convolution Time: %lf seconds\n", time);

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

	printf("Tiled Convolution Output Checksum: %lf\n", checksum);

	free(h_inp);
	free(h_filt);
	free(h_out);
	cudaFree(d_inp);
	cudaFree(d_filt);
	cudaFree(d_out);
	
	return 0;
}