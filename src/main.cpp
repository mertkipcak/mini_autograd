#include "tensor.hpp"
#include "functional.hpp"
#include <iostream>
#include <vector>
#include <chrono>

void test_accuracy() {
     // Tensor A: shape (2, 3)
     std::vector<float> a_data = {
        1, 2, 3,
        4, 5, 6
    };
    std::vector<int> a_shape = {2, 3};
    Tensor A(a_data, a_shape);
    std::cout << A.to_string() << std::endl;


    // Tensor B: shape (3, 2)
    std::vector<float> b_data = {
        7, 8,
        9, 10,
        11, 12
    };
    std::vector<int> b_shape = {2, 3};
    Tensor B(b_data, b_shape);
    std::cout << B.to_string() << std::endl;

    Tensor D = B.transpose();
    std::cout << D.to_string() << std::endl;
    Tensor C = matmul(A, D);
    std::cout << C.to_string() << std::endl;
}

void test_speed() {
    // Use large shapes for benchmarking
    Tensor A = randn({2048, 2048});
    Tensor B = randn({2048, 2048});

    // Contiguous matmul
    auto start1 = std::chrono::high_resolution_clock::now();
    Tensor C1 = matmul(A, B);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Contiguous matmul time: " << elapsed1.count() << " seconds" << std::endl;

    // Non-contiguous matmul
    auto start2 = std::chrono::high_resolution_clock::now();
    Tensor C2 = matmul(A, B.transpose());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Contiguous matmul time: " << elapsed2.count() << " seconds" << std::endl;
}

int main() {
    test_speed();

    return 0;
}