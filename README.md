# GPU/CUDA Optimizations

This project is about benchmarking operations like vector addition, matrix multiplications, tiling, and convolution in a GPU via CUDA programming and cuDNN and on CPU using C/C++. There are 3 components:

- `vecadd-matmull/`: This folder contains CUDA programs for naive vector addition, vector addition using coalesced memory access, naive matrix multiplication, matrix multiplication using shared memory, and benchmarks.
- `unified-mem/`: This folder contains programs for naive vector addition in C++, vector addition using unified memory in CUDA, and benchmarks comparing the two implementations.
- `convolution/`: This folder contains CUDA programs for naive convolution, tiled convolution, and convolution using cuDNN.
