#include "tensor.hpp"
#include "functional.hpp"
#include <iostream>
#include <vector>

int main() {
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
    std::vector<int> b_shape = {3, 2};
    Tensor B(b_data, b_shape);
    std::cout << B.to_string() << std::endl;

    Tensor C = apply_unary(A, [](float x) { return x + 2.0; });
    std::cout << C.to_string() << std::endl;

    return 0;
}
