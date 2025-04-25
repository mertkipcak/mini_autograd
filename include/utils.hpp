
#pragma once

#include "types.hpp"
#include <vector>
#include <cmath>
#include <optional>

/**
 * @brief Calculate the total number of elements from a shape vector
 * @param shape A vector representing the dimensions of a tensor
 * @return The total number of elements in a tensor with the given shape
 */
inline size_t numel_shape(const t_shape& shape) {
    size_t size = 1;
    for(size_t i = 0; i < shape.size(); i++) {
        size *= (size_t) shape.at(i);
    }
    return size;
}

/**
 * @brief Get the broadcasted shape of two tensors, if possible
 * @param a First tensor
 * @param b Second tensor
 * @return The new broadcasted shape or std::nullopt if incompatible
 */
std::optional<t_shape> broadcast_shape(const t_shape& a, const t_shape& b);

/**
 * @brief Prepends 1s to the beginning of a shape until it reaches the desired size
 * @param shape The original shape vector
 * @param size The desired size of the output shape
 * @return A new shape vector with 1s prepended if needed
 */
inline t_shape pad_shape_to_size(const t_shape& shape, size_t size, size_t pad = 1) {
    if (shape.size() >= size) {
        return shape;
    }
    t_shape padded(size, pad);
    std::copy(shape.begin(), shape.end(), padded.end() - shape.size());
    return padded;
}

/**
 * @brief Compute the sigmoid function for a single value
 * @param x Input value
 * @return Sigmoid result: 1/(1+e^(-x))
 */
inline float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}
