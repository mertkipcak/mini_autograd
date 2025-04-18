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

    // Tensor B: shape (3, 2)
    std::vector<float> b_data = {
        7, 8,
        9, 10,
        11, 12
    };
    std::vector<int> b_shape = {3, 2};
    Tensor B(b_data, b_shape);

    Tensor C = dot(A, B); // Should be shape (2, 2)

    std::cout << "Result of A.dot(B):\n";
    for (size_t i = 0; i < C.numel(); ++i) {
        std::cout << C[i] << " ";
        if ((i + 1) % 2 == 0) std::cout << std::endl;
    }

    return 0;
}
