#include "tensor.hpp"
#include "functional.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Use large shapes for benchmarking
    Tensor A = randn({2048, 2048});
    Tensor B = randn({2048, 2048});

    // Contiguous matmul
    auto start1 = std::chrono::high_resolution_clock::now();
    Tensor C1 = dot(A, B);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Contiguous matmul time: " << elapsed1.count() << " seconds" << std::endl;

    // Non-contiguous matmul (A is transposed)
    auto start2 = std::chrono::high_resolution_clock::now();
    Tensor C2 = dot(A.transpose(), B);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Non-contiguous (transposed) matmul time: " << elapsed2.count() << " seconds" << std::endl;

    return 0;
}