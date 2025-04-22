#pragma once

#include "tensor.hpp"

/**
 * @brief Create a new tensor from the matrix multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor matmul(const t_tensor& a, const t_tensor& b);