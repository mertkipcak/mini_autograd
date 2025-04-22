#pragma once

#include "tensor.hpp"
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
t_tensor randn(const t_shape& shape, bool requires_grad=false);

/**
 * @brief sum the elements of the matrics, across a dimension if given
 */
t_tensor sum(const t_tensor& input, int dim = -1, bool keepdims = false);
