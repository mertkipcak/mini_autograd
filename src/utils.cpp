
#include "utils.hpp"
#include <cassert>

size_t numel_shape(const t_shape& shape) {
    size_t size = 1;
    for(size_t i = 0; i < shape.size(); i++) {
        size *= (size_t) shape.at(i);
    }
    return size;
}

std::optional<t_shape> broadcast_shape(const t_shape& a, const t_shape& b) {
    // Scalar check
    if (!a.size()) return b;
    if (!b.size()) return a;

    // Setup
    size_t dim_a = a.size();
    size_t dim_b = b.size();
    size_t max_dim = dim_a > dim_b ? dim_a : dim_b;
    t_shape shape_a = t_shape(max_dim, 1);
    t_shape shape_b = t_shape(max_dim, 1);
    t_shape broadcasted_shape(max_dim);

    // Apply broadcast
    for(size_t i = max_dim; i-- > 0;) {
        if (i + dim_a >= max_dim)
            shape_a[i] = a[i - max_dim + dim_a];
        
        if (i + dim_b >= max_dim)
            shape_b[i] = b[i - max_dim + dim_b];

        if (shape_a[i] == shape_b[i]) {
            broadcasted_shape[i] = shape_a[i];
            continue;
        } else if (shape_a[i] == 1) {
            broadcasted_shape[i] = shape_b[i];
            continue;
        } else if (shape_b[i] == 1) {
            broadcasted_shape[i] = shape_a[i];
            continue;
        }
        return std::nullopt;
    }

    return broadcasted_shape;
}

t_shape pad_shape_to_size(const t_shape& shape, size_t size, size_t pad) {
    if (shape.size() >= size) {
        return shape;
    }
    t_shape padded(size, pad);
    std::copy(shape.begin(), shape.end(), padded.end() - shape.size());
    return padded;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

