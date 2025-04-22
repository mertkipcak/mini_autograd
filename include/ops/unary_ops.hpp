#pragma once

#include "tensor.hpp"

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