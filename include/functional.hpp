#pragma once

#include "tensor.hpp"
#include "tensor_iterator.hpp"
#include "utils.hpp"

/**
 * @brief Check if two tensors have the same shape
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return True if shapes match
 */
bool same_shape(const Tensor& t1, const Tensor& t2);

/**
 * @brief Get the broadcasted shape if 
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return The new shape
 */
std::optional<t_shape> broadcast_shape(const Tensor& t1, const Tensor& t2);

/**
 * @brief Broadcast the data of a tensor to a new shape.
 * @param t The tensor whose data will be broadcasted
 * @param broadcast_shape The shape to which the data will be broadcasted
 * @return The broadcasted data as a t_data object
 */
t_data broadcast_data(const Tensor& t, t_shape broadcast_shape);

/**
 * @brief Apply a binary operation element-wise to two tensors.
 * @param t1 First tensor
 * @param t2 Second tensor
 * @param op The binary operator to apply element-wise
 * @return New tensor containing the result of the operation
 */
Tensor apply_binary(const Tensor& t1, const Tensor& t2, std::function<float(float, float)> op);


/**
 * @brief Create a new tensor that is the element-wise sum of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor add(const Tensor& t1, const Tensor& t2);

/**
 * @brief Create a new tensor that is the element-wise product of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor mul(const Tensor& t1, const Tensor& t2);

/**
 * @brief Create a new tensor from the matrix multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
Tensor dot(const Tensor& t1, const Tensor& t2);

/**
 * @brief Apply an opration element-wise to a tensor
 * @param a Input tensor
 * @param op the operator to apply element-wise
 * @return New tensor with opeartion applied
 */
Tensor apply_unary(const Tensor& t, std::function<float(float)> op);

/**
 * @brief Apply sigmoid function element-wise to a tensor
 * @param a Input tensor
 * @return New tensor with sigmoid applied
 */
Tensor sigmoid(const Tensor& t);

/**
 * @brief Apply exponential function element-wise to a tensor
 * @param a Input tensor
 * @return New tensor with exp applied
 */
Tensor exp(const Tensor& t);

/**
 * @brief Apply natural logarithm element-wise to a tensor
 * @param a Input tensor
 * @return New tensor with log applied
 */
Tensor log(const Tensor& t);
