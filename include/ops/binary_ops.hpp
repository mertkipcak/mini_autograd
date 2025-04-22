#pragma once

#include "tensor.hpp"

/**
 * @brief Apply a binary operation element-wise to two tensors.
 * @param a First tensor
 * @param b Second tensor
 * @param op The binary operator to apply element-wise
 * @return New tensor containing the result of the operation
 */
t_tensor apply_binary(const t_tensor& a, const t_tensor& b, std::function<float(float, float)> op);

/**
 * @brief Create a new tensor that is the element-wise sum of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor add(const t_tensor& a, const t_tensor& b);

/**
 * @brief Create a new tensor that is the element-wise product of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor mul(const t_tensor& a, const t_tensor& b);