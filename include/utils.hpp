
#pragma once

#include "types.hpp"
#include <vector>
#include <cmath>

/**
 * @brief Calculate the total number of elements from a shape vector
 * @param shape A vector representing the dimensions of a tensor
 * @return The total number of elements in a tensor with the given shape
 */
int numel_shape(const t_shape& shape);

/**
 * @brief Prepends 1s to the beginning of a shape until it reaches the desired size
 * @param shape The original shape vector
 * @param size The desired size of the output shape
 * @return A new shape vector with 1s prepended if needed
 */
t_shape pad_shape_to_size(const t_shape& shape, size_t size, int pad = 1);

/**
 * @brief Compute the sigmoid function for a single value
 * @param x Input value
 * @return Sigmoid result: 1/(1+e^(-x))
 */
float sigmoid(float x);