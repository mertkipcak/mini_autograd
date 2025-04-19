#include "functional.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>


bool same_shape(const Tensor& t1, const Tensor& t2) {
    return t1.get_shape() == t2.get_shape();
}

std::optional<t_shape> broadcast_shape(const Tensor& t1, const Tensor& t2) {
    // Setup
    size_t dim1 = t1.get_shape().size();
    size_t dim2 = t2.get_shape().size();
    size_t max_dim = dim1 > dim2 ? dim1 : dim2;
    t_shape s1 = t_shape(max_dim, 1);
    t_shape s2 = t_shape(max_dim, 1);
    t_shape broadcasted_shape(max_dim);

    // Apply broadcast
    for(size_t i = max_dim; i-- > 0;) {
        if (i + dim1 >= max_dim)
            s1[i] = t1.get_shape()[i - max_dim + dim1];
        
        if (i + dim2 >= max_dim)
            s2[i] = t2.get_shape()[i - max_dim + dim2];

        if (s1[i] == s2[i]) {
            broadcasted_shape[i] = s1[i];
            continue;
        } else if (s1[i] == 1) {
            broadcasted_shape[i] = s2[i];
            continue;
        } else if (s2[i] == 1) {
            broadcasted_shape[i] = s1[i];
            continue;
        }
        return std::nullopt;
    }

    return broadcasted_shape;
}

Tensor apply_binary(const Tensor& t1, const Tensor& t2, std::function<float(float, float)> op) {
    std::optional<t_shape> maybe_shape = broadcast_shape(t1, t2);
    if (!maybe_shape.has_value()) throw std::runtime_error("Shape mismatch at Tensor binary operations");

    t_shape shape = maybe_shape.value();

}

Tensor add(const Tensor& t1, const Tensor& t2) {
    return apply_binary(t1, t2, [](float x, float y) { return x + y; });
}

Tensor mul(const Tensor& t1, const Tensor& t2) {
    return apply_binary(t1, t2, [](float x, float y) { return x * y; });
}

Tensor dot(const Tensor& t1, const Tensor& t2) {
    // Assertions and setup
    assert(!t1.get_shape().empty() && !t2.get_shape().empty());
    assert(t1.get_shape().back() == t2.get_shape().front());
    
    t_shape new_shape = t_shape();
    for(size_t i = 0; i < t1.get_shape().size() - 1; i++) 
        new_shape.push_back(t1.get_shape()[i]);
    for(size_t i = 1; i < t2.get_shape().size(); i++)
        new_shape.push_back(t2.get_shape()[i]);

    int new_size = numel_shape(new_shape);
    t_data new_data(new_size);
    bool required_grad = t1.get_requires_grad() || t2.get_requires_grad();
    Tensor result = Tensor(new_data, new_shape, required_grad);

    // Actual matmul

    // result.at({...a, ...b}) = sum()
    return result;
}

Tensor apply_unary(const Tensor& t, std::function<float(float)> op) {
    t_data data(t.get_data());

    for(size_t i = 0; i < data.size(); i++) {
        data[i] = op(data[i]);
    }

    return Tensor(
        data,
        std::vector<int>(t.get_shape()),
        t.get_requires_grad()
    );
}

Tensor sigmoid(const Tensor& t) {
    return apply_unary(t, [](float x) { return sigmoid(x); });
}

Tensor exp(const Tensor& t) {
    return apply_unary(t, [](float x) { return exp(x); });
}

Tensor log(const Tensor& t) {
    return apply_unary(t, [](float x) { return log(x); });
}
