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
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return True if shapes match
 */
bool same_shape(const Tensor& t1, const Tensor& t2);

/**
 * @brief Get the broadcasted shape of two tensors, if possible
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return The new broadcasted shape or std::nullopt if incompatible
 */
std::optional<t_shape> broadcast_shape(const Tensor& t1, const Tensor& t2);

/**
 * @brief Apply a binary operation element-wise to two tensors.
 * @param t1 First tensor
 * @param t2 Second tensor
 * @param op The binary operator to apply element-wise
 * @return New tensor containing the result of the operation
 */
Tensor apply_binary(const Tensor& t1, const Tensor& t2, std::function<float(float, float)> op);

/**
 * @brief Apply an operation element-wise to a tensor
 * @param a Input tensor
 * @param op the operator to apply element-wise
 * @return New tensor with operation applied
 */
Tensor apply_unary(const Tensor& t, std::function<float(float)> op);

/**
 * @brief Create a new tensor from the matrix multiplication of two tensors
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return New tensor containing the result
 */
Tensor dot(const Tensor& t1, const Tensor& t2);

/**
 * @brief Create a new tensor that is the element-wise sum of two tensors
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return New tensor containing the result
 */
Tensor add(const Tensor& t1, const Tensor& t2);

/**
 * @brief Create a new tensor that is the element-wise product of two tensors
 * @param t1 First tensor
 * @param t2 Second tensor
 * @return New tensor containing the result
 */
Tensor mul(const Tensor& t1, const Tensor& t2);

/**
 * @brief Apply sigmoid function element-wise to a tensor
 * @param t Input tensor
 * @return New tensor with sigmoid applied
 */
Tensor sigmoid(const Tensor& t);

/**
 * @brief Apply exponential function element-wise to a tensor
 * @param t Input tensor
 * @return New tensor with exp applied
 */
Tensor exp(const Tensor& t);

/**
 * @brief Apply natural logarithm element-wise to a tensor
 * @param t Input tensor
 * @return New tensor with log applied
 */
Tensor log(const Tensor& t);

/**
 * @brief Apply softmax function over the specified axis
 * @param t Input tensor
 * @param axis Axis to apply softmax along (default -1 = last dim)
 * @return New tensor with softmax probabilities
 */
Tensor softmax(const Tensor& t, int axis = -1);

/**
 * @brief Apply layer normalization over the last dimension
 * @param t Input tensor
 * @param eps Small constant for numerical stability
 * @return New tensor with normalized values
 */
Tensor layernorm(const Tensor& t, float eps = 1e-5);

/**
 * @brief Apply dropout to the input tensor
 * @param t Input tensor
 * @param p Dropout probability
 * @param training Whether to apply dropout (false = pass through)
 * @return New tensor with dropped elements
 */
Tensor dropout(const Tensor& t, float p, bool training = true);

/**
 * @brief Look up embeddings by indices
 * @param indices Tensor of indices (ints stored as floats)
 * @param weight Embedding weight matrix (num_embeddings x dim)
 * @return New tensor containing embedded vectors
 */
Tensor embedding(const Tensor& indices, const Tensor& weight);

/**
 * @brief Concatenate tensors along the specified axis
 * @param tensors Vector of tensors with the same shape except along axis
 * @param axis Axis along which to concatenate
 * @return New concatenated tensor
 */
Tensor concat(const std::vector<Tensor>& tensors, int axis);

/**
 * @brief Split tensor into equal chunks along axis
 * @param t Input tensor
 * @param chunks Number of chunks
 * @param axis Axis along which to split
 * @return Vector of chunked tensors
 */
std::vector<Tensor> split(const Tensor& t, int chunks, int axis);
