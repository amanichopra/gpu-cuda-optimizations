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

// Kernel for 2D convolution
__global__ void untiled_convolution1(double *image, double *filters, double *output, int image_width, int image_height, int num_filters, int filter_size) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int filter_x, filter_y, image_x, image_y;
    double sum;

    int filter_index = by * gridDim.x + bx;

    if (filter_index < num_filters) {
        sum = 0.0f;
        for (int c = 0; c < 3; ++c) { // Iterate over RGB channels
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    filter_x = bx * blockDim.x + tx - filter_size / 2 + j;
                    filter_y = by * blockDim.y + ty - filter_size / 2 + i;

                    image_x = filter_x;
                    image_y = filter_y;

                    // Check boundary conditions
                    if (image_x >= 0 && image_x < image_width && image_y >= 0 && image_y < image_height) {
                        sum += filters[filter_index * filter_size * filter_size * 3 + i * filter_size * 3 + j * 3 + c] *
                            image[(image_y * image_width + image_x) * 3 + c];
                    }
                }
            }
        }
        output[filter_index * image_width * image_height + (image_height * by + ty) * image_width + (bx * blockDim.x + tx)] = sum;
    }
}

__global__ void untiled_convolution(double *inp, int H, int W, int C, double *filter, int FH, int FW, int K, double *out, int block_size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
	int twid = block_size + (FW / 2) * 2;
	int thi = block_size + (FH / 2) * 2;
	extern __shared__ double ti[];
	if (i < H && j < W && k < K) {
        int t_i = threadIdx.x + FH / 2;
		int t_j = threadIdx.y + FW / 2;
		for (int c = 0; c < C; ++c) {
			ti[t_j + (t_i * twid) + (c * thi * twid)] = inp[j + (i * W) + (c * H * W)];
			if (threadIdx.x == 0) {
				if (i > 0) {
					ti[t_j + (t_i - 1) * twid + c * thi * twid] = inp[j + (i - 1) * W + c * H * W];
				}
				else {
					ti[t_j + (t_i - 1) * twid + c * thi * twid] = 0.0;
				}

				if (threadIdx.y == 0) {
					if (j > 0 && i > 0) {
						ti[(t_j - 1) + (t_i - 1) * twid + c * thi * twid] = inp[(j - 1) + (i - 1) * W + c * H * W];
					}
					else {
						ti[(t_j - 1) + (t_i - 1) * twid + c * thi * twid] = 0.0;
					}

				}
			}
			if (threadIdx.y == block_size - 1) {
				if (j > W - 1) {
                    ti[(t_j + 1) + t_i * twid + c * thi * twid] = 0.0;
					ti[(t_j + 1) + t_i * twid + c * thi * twid] = inp[(j + 1) + i * W + c * H * W];
				}

				else {
					ti[(t_j + 1) + t_i * twid + c * thi * twid] = inp[(j + 1) + i * W + c * H * W];
				}

				if (threadIdx.x == 0) {
					if (j < W - 1 && i > 0) {
						ti[(t_j + 1) + (t_i - 1) * twid + c * thi * twid] = inp[(j + 1) + (i - 1) * W + c * H * W];
					}
					else {
						ti[(t_j + 1) + (t_i - 1) * twid + c * thi * twid] = 0.0;
					}
				}
			}
			if (threadIdx.x == block_size - 1) {
				if (i < H - 1) {
					ti[t_j + (t_i + 1) * twid + c * thi * twid] =   inp[j + (i + 1) * W + c * H * W];
				}
				else {
					ti[t_j + (t_i + 1) * twid + c * thi * twid] = 0.0;
				}
				if (threadIdx.y == block_size - 1) {
					if (j < W - 1 && i < H - 1)
					{
						ti[(t_j + 1) + (t_i + 1) * twid + c * thi * twid] =  inp[(j + 1) + (i + 1) * W + c * H * W];
					}
					else {
						ti[(t_j + 1) + (t_i + 1) * twid + c * thi * twid] = 0.0;

					}
				}
			}
			if (threadIdx.y == 0) {
				if (j > 0) {
					ti[(t_j - 1) + t_i * twid + c * thi * twid] =  inp[(j - 1) + i * W + c * H * W];
				}
				else {
					ti[(t_j - 1) + t_i * twid + c * thi * twid] = 0.0;
				}
				if (threadIdx.x == block_size - 1) {
					if (j > 0 && i < H - 1) {
						ti[(t_j - 1) + (t_i + 1) * twid + c * thi * twid] = inp[(j - 1) + (i + 1) * W + c * H * W];
					}
					else {
						ti[(t_j - 1) + (t_i + 1) * twid + c * thi * twid] = 0.0;
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

    struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
	untiled_convolution<<<num_blocks, num_threads, (BLOCK_SIZE + (FH / 2) * 2) * (BLOCK_SIZE + (FW / 2) * 2) * C * sizeof(double)>>>(d_inp, H, W, C, d_filt, FH, FW, K, d_out, BLOCK_SIZE);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	double time = (end.tv_sec - start.tv_sec) +(end.tv_nsec - start.tv_nsec) / 1E9;
    printf("Untiled Convolution Time: %lf seconds\n", time);

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

	printf("Untiled Convolution Output Checksum: %lf\n", checksum);

	free(h_inp);
	free(h_filt);
	free(h_out);
	cudaFree(d_inp);
	cudaFree(d_filt);
	cudaFree(d_out);
	
	return 0;
}

