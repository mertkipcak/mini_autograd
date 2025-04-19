#include "tensor.hpp"
#include "functional.hpp"
#include <iostream>
#include <vector>

int main() {
    Tensor A = randn({3, 4, 3});
    Tensor B = randn({4, 2});

    Tensor C = dot(A.transpose(), B);

    std::cout << A.to_string() << std::endl;
    std::cout << B.to_string() << std::endl;
    std::cout << C.to_string() << std::endl;

    return 0;
}
