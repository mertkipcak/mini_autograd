#pragma once

#include "tensor.hpp"
#include <functional>
#include <optional>
#include <vector>
#include <random>

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
 * @brief Create a new tensor that is the element-wise sub of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result with elements a - b
 */
t_tensor sub(const t_tensor& a, const t_tensor& b);

/**
 * @brief Create a new tensor that is the element-wise product of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor mul(const t_tensor& a, const t_tensor& b);

/**
 * @brief Create a new tensor that is the element-wise division of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor div(const t_tensor& a, const t_tensor& b);

/**
 * @brief Create a new tensor from the tensor multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor matmul(const t_tensor& a, const t_tensor& b);

/**
 * @brief Get a tensor with a given shape initialized with normal random weights
 * @param shape Shape of the return tensor
 * @param requires_grad Wheter to track gradients or not
 * @return Random tensor
 */
t_tensor randn(const t_shape& shape, bool requires_grad=false);

/**
 * @brief sum the elements of the tensor, across a given dimension
 */
t_tensor sum(const t_tensor& input, size_t dim, bool keepdims = false);

/**
 * @brief sum all the elements of the tensor
 */
t_tensor sumall(const t_tensor& input);

/**
 * @brief Apply an operation element-wise to a tensor
 * @param input Input tensor
 * @param op the operator to apply element-wise
 * @return New tensor with operation applied
 */
t_tensor apply_unary(const t_tensor& input, std::function<float(float)> op);

/**
 * @brief apply sigmoid function element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with sigmoid applied
 */
t_tensor sigmoid(const t_tensor& input);

/**
 * @brief apply exponential function element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with exp applied
 */
t_tensor exp(const t_tensor& input);

/**
 * @brief apply natural logarithm element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with log applied
 */
t_tensor log(const t_tensor& input);

/**
 * @brief square element-wise a tensor
 * @param input Input tensor
 * @return New squared tensor
 */
t_tensor square(const t_tensor& input);