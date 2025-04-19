
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
 * @brief Compute the sigmoid function for a single value
 * @param x Input value
 * @return Sigmoid result: 1/(1+e^(-x))
 */
float sigmoid(float x);