
#include "utils.hpp"
#include <cassert>

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
