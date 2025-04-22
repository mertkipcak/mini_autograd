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
    std::vector<int> a_shape = {1, 6};
    t_tensor A = create_tensor(a_data, a_shape);
    std::cout << A->to_string() << std::endl;


    // Tensor B: shape (3, 2)
    std::vector<float> b_data = {
        7, 8,
        9, 10,
        11, 12
    };
    std::vector<int> b_shape = {6};
    t_tensor B = create_tensor(b_data, b_shape);
    std::cout << B->to_string() << std::endl;

    t_tensor C = matmul(A, B);
    std::cout << C->to_string() << std::endl;
}

void test_unary_speed() {
    t_tensor A = randn({4096, 4096, 8});

    auto start1 = std::chrono::high_resolution_clock::now();
    t_tensor B = exp(A);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Optimized time: " << elapsed1.count() << " seconds" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    t_tensor C = apply_binary(A->transpose(), A->transpose(), [](float x, float y){ return exp(x); });
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Non-optimized time: " << elapsed2.count() << " seconds" << std::endl;
}

void test_speed() {
    // Use large shapes for benchmarking
    t_tensor A = randn({2048, 2048});
    t_tensor B = randn({2048, 2048});

    // Contiguous matmul
    auto start1 = std::chrono::high_resolution_clock::now();
    t_tensor C1 = matmul(A, B);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Contiguous matmul time: " << elapsed1.count() << " seconds" << std::endl;

    // Non-contiguous matmul
    auto start2 = std::chrono::high_resolution_clock::now();
    t_tensor C2 = matmul(A, B->transpose());
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start2;
    std::cout << "Non-contiguous matmul time: " << elapsed2.count() << " seconds" << std::endl;
}

void test_backprop() {
    t_tensor A = create_tensor(t_data({1}), t_shape({1, 1}), true);
    t_tensor B = create_tensor(t_data({2}), t_shape({1, 1}), true);
    t_tensor C = create_tensor(t_data({3}), t_shape({1, 1}), true);
    t_tensor F = add(A, mul(B, C));

    F->backward();

    std::cout << A->to_string() << std::endl;
    std::cout << B->to_string() << std::endl;
    std::cout << C->to_string() << std::endl;
    std::cout << F->to_string() << std::endl;
}

int main() {
    test_backprop();
    return 0;
}