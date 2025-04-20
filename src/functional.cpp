#include "functional.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>

Tensor randn(const t_shape& shape, bool requires_grad) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t numel = numel_shape(shape);
    t_data data(numel);

    for (size_t i = 0; i < numel; ++i) data[i] = dist(gen);

    return Tensor(data, shape, requires_grad);
}

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

    Tensor res(t_data(numel_shape(shape)), shape, t1.get_requires_grad() || t2.get_requires_grad());

    for(TensorIterator it(shape); !it.done(); it.inc()) {
        t_indices indices = it.get();
        res.at(indices) = op(t1.broadcasted_at(indices, shape), t2.broadcasted_at(indices, shape));
    }

    return res;
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

void dot_contiguous(const Tensor& t1, const Tensor& t2, Tensor& res) {
    const t_data& d1 = t1.get_data();
    const t_data& d2 = t2.get_data();
    t_data& dr = res.get_data();

    t_shape s1 = t1.get_shape();
    t_shape s2 = t2.get_shape();
    t_shape sr = res.get_shape();

    const t_shape& lhs_strides = t1.get_strides();
    const t_shape& rhs_strides = t2.get_strides();

    for (TensorIterator it(sr); !it.done(); it.inc()) {
        t_indices indices = it.get();
        int lhs_flat = 0, rhs_flat = 0;
        for (size_t d = 0; d < s1.size() - 1; ++d)
            lhs_flat += indices[d] * lhs_strides[d];
        for (size_t d = 1; d < s2.size(); ++d)
            rhs_flat += indices[s1.size() - 1 + d - 1] * rhs_strides[d];

        float value = 0.0f;
        for (int k = 0; k < s1.back(); ++k)
            value += d1[lhs_flat + k * lhs_strides[s1.size() - 1]] *
                        d2[rhs_flat + k * rhs_strides[0]];

        dr[it.get_offset()] = value;
    }
}

void dot_generic(const Tensor& t1, const Tensor& t2, Tensor& res) {
    t_shape s1 = t1.get_shape();
    t_shape s2 = t2.get_shape();
    t_shape lhs_index(s1.size());
    t_shape rhs_index(s2.size());

    for(TensorIterator it(res.get_shape()); !it.done(); it.inc()) {
        t_indices indices = it.get();
        t_indices left_indices = indices.subspan(0, s1.size() - 1);
        t_indices right_indices = indices.subspan(s1.size() - 1);
        std::copy(left_indices.begin(), left_indices.end(), lhs_index.begin());
        std::copy(right_indices.begin(), right_indices.end(), rhs_index.begin() + 1);

        float value = 0;

        for(size_t i = 0; i < s1.back(); i++) {
            lhs_index[s1.size() - 1] = i;
            rhs_index[0] = i;

            t_indices li(lhs_index);
            t_indices ri(rhs_index);

            value += t1.at(li) * t2.at(ri);
        }

        res.at(indices) = value;
    }
}

Tensor dot(const Tensor& t1, const Tensor& t2) {
    // Assertions and setup
    t_shape s1 = t1.get_shape();
    t_shape s2 = t2.get_shape();

    assert(!s1.empty() && !s2.empty());
    assert(s1.back() == s2.front());
    
    t_shape new_shape;
    new_shape.reserve(s1.size() + s2.size() - 2);
    new_shape.insert(new_shape.end(), s1.begin(), s1.end() - 1);
    new_shape.insert(new_shape.end(), s2.begin() + 1, s2.end());

    int new_size = numel_shape(new_shape);
    t_data new_data(new_size, 0.0f);
    bool required_grad = t1.get_requires_grad() || t2.get_requires_grad();
    Tensor res(new_data, new_shape, required_grad);

    // Matmul
    if (t1.get_contiguous() && t2.get_contiguous()) {
        dot_contiguous(t1, t2, res);
    } else {
        dot_generic(t1, t2, res);
    }
    
    return res;
}

Tensor add(const Tensor& t1, const Tensor& t2) {
    return apply_binary(t1, t2, [](float x, float y) { return x + y; });
}

Tensor mul(const Tensor& t1, const Tensor& t2) {
    return apply_binary(t1, t2, [](float x, float y) { return x * y; });
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
