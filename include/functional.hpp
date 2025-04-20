#pragma once

#include "tensor.hpp"
#include "tensor_iterator.hpp"
#include "utils.hpp"
#include <functional>
#include <optional>
#include <vector>
#include <random>

/**
 * @brief Get a matrix with a given shape initialized with normal random weights
 * @param shape Shape of the return matrix
 * @param requires_grad Wheter to track gradients or not
 * @return Random matrix
 */
Tensor randn(const t_shape& shape, bool requires_grad=false);

/**
 * @brief Check if two tensors have the same shape
 * @param a First tensor
 * @param b Second tensor
 * @return True if shapes match
 */
bool same_shape(const Tensor& a, const Tensor& b);

/**
 * @brief Get the broadcasted shape of two tensors, if possible
 * @param a First tensor
 * @param b Second tensor
 * @return The new broadcasted shape or std::nullopt if incompatible
 */
std::optional<t_shape> broadcast_shape(const Tensor& a, const Tensor& b);

/**
 * @brief Apply a binary operation element-wise to two tensors.
 * @param a First tensor
 * @param b Second tensor
 * @param op The binary operator to apply element-wise
 * @return New tensor containing the result of the operation
 */
Tensor apply_binary(const Tensor& a, const Tensor& b, std::function<float(float, float)> op);

/**
 * @brief Apply an operation element-wise to a tensor
 * @param input Input tensor
 * @param op the operator to apply element-wise
 * @return New tensor with operation applied
 */
Tensor apply_unary(const Tensor& input, std::function<float(float)> op);

/**
 * @brief Create a new tensor from the matrix multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Create a new tensor that is the element-wise sum of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Create a new tensor that is the element-wise product of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief apply sigmoid function element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with sigmoid applied
 */
Tensor sigmoid(const Tensor& input);

/**
 * @brief apply exponential function element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with exp applied
 */
Tensor exp(const Tensor& input);

/**
 * @brief apply natural logarithm element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with log applied
 */
Tensor log(const Tensor& input);
