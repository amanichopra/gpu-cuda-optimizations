#include <iostream>
#include <chrono>
#include <cstdlib>

void addArrays(float *A, float *B, float *C, int K) {
    for (int i = 0; i < K * 1000000; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <K_value>" << std::endl;
        return 1;
    }

    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cout << "Please provide a valid positive value for K." << std::endl;
        return 1;
    }

    int size = K * 1000000;
    float *A = new float[size];
    float *B = new float[size];
    float *C = new float[size];

    // Initialize arrays A and B (for simplicity, just assigning some values)
    for (int i = 0; i < size; ++i) {
        A[i] = i * 1.5;
        B[i] = i * 2.5;
    }

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform vector addition
    addArrays(A, B, C, K);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken for K = " << K << ": " << duration.count() << " seconds" << std::endl;

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}