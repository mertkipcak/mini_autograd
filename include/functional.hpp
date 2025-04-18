#pragma once

#include "tensor.hpp"
#include "utils.hpp"

/**
 * @brief Check if two tensors have the same shape
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return True if shapes match
 */
bool same_shape(const Tensor& t1, const Tensor& t2);

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
 * @brief Create a new tensor from the matrix multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor dot(const Tensor& a, const Tensor& b);

/**
 * @brief Apply sigmoid function element-wise to a tensor
 * @param a Input tensor
 * @return New tensor with sigmoid applied
 */
Tensor sigmoid(const Tensor& a);

/**
 * @brief Apply exponential function element-wise to a tensor
 * @param a Input tensor
 * @return New tensor with exp applied
 */
Tensor exp(const Tensor& a);

/**
 * @brief Apply natural logarithm element-wise to a tensor
 * @param a Input tensor
 * @return New tensor with log applied
 */
Tensor log(const Tensor& a);
