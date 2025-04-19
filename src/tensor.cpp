#include "tensor.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

Tensor::Tensor(
    const t_data& data_,
    const t_shape& shape_,
    bool requires_grad
) : data(data_), shape(shape_), requires_grad(requires_grad) {
    assert(numel_shape(shape) == data.size());

    // Setup strides
    strides = t_strides(shape.size(), 1);
    for(int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }

    // Setup grad
    if (requires_grad) {
        grad = t_data(data.size(), 0.0f);
    }
}

Tensor::Tensor(
    const t_data& data_,
    const t_shape& shape_,
    const t_shape& strides_,
    bool requires_grad
) : data(data_), shape(shape_), strides(strides_), requires_grad(requires_grad) {
    assert(numel_shape(shape) == data.size());

    // Setup strides
    strides = t_strides(shape.size(), 1);
    for(int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }

    // Setup grad
    if (requires_grad) {
        grad = t_data(data.size(), 0.0f);
    }
}

const t_shape& Tensor::get_shape() const {
    return shape;
}

const t_data& Tensor::get_data() const {
    return data;
}

void Tensor::set_data(const t_data& new_data) {
    assert(new_data.size() == data.size() && "Data size must match the shape size!");
    data = new_data;
}

bool Tensor::get_requires_grad() const {
    return requires_grad;
}

void Tensor::set_requires_grad(bool new_requires_grad) {
    requires_grad = new_requires_grad;

    if (requires_grad && grad.size() != data.size()) {
        grad.resize(data.size(), 0.0f);
    }
}

float& Tensor::at(t_indices& indices) {
    assert(shape.size() == indices.size());

    int flat_index = 0;
    for(int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0 && indices[i] < shape[i]);
        flat_index += indices[i] * strides[i];
    }

    return data.at(flat_index);
}

const float& Tensor::at(t_indices& indices) const {
    assert(shape.size() == indices.size());

    int flat_index = 0;
    for(int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0 && indices[i] < shape[i]);
        flat_index += indices[i] * strides[i];
    }

    return data.at(flat_index);
}

const float& Tensor::broadcasted_at(const t_indices& indices, const t_shape& broadcasted_shape) const {
    int flat_index = 0;
    for(size_t i = 1; i <= shape.size(); i ++) {
        size_t bo = broadcasted_shape.size() - i; // broadcasted offset
        size_t o = shape.size() - i; // offset
        assert(broadcasted_shape[bo] == shape[o] || shape[o] == 1);
        if (shape[o] == 1) continue;
        flat_index += indices[bo] * strides[o];
    }

    return data.at(flat_index);
}

bool Tensor::is_contiguous() const {
    int stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {  
        if (strides[i] != stride) return false;
        stride *= shape[i];
    }

    return true;
}

void Tensor::backward() {}

void Tensor::zero_grad() {
    grad.assign(data.size(), 0.0f);
}

size_t Tensor::numel() const {
    return data.size();
}

std::string Tensor::to_string() const {
    size_t limit = 50;
    std::stringstream ss;

    ss << "Tensor(";
    ss << "shape=";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i != shape.size() - 1) {
            ss << "x";
        }
    }

    ss << ", requires_grad=" << (requires_grad ? "True" : "False");

    ss << ", data=[";
    size_t preview_size = std::min(data.size(), limit);
    for (size_t i = 0; i < preview_size; ++i) {
        ss << data[i];
        if (i != preview_size - 1) {
            ss << ", ";
        }
    }

    if (data.size() > limit) {
        ss << ", ...";
    }

    ss << "])";
    
    return ss.str();
}

Tensor Tensor::transpose() const {
    if (shape.size() == 1) {
        t_shape new_shape = {1, shape[0]};
        t_shape new_strides = {shape[0], 1};
        t_data new_data(data);
        return Tensor(new_data, new_shape, requires_grad);
    }

    t_shape new_shape(shape);
    std::swap(new_shape[new_shape.size() - 2], new_shape[new_shape.size() - 1]);

    t_shape new_strides(strides);
    std::swap(new_strides[new_strides.size() - 2], new_strides[new_strides.size() - 1]);
    t_data new_data(data);
    return Tensor(new_data, new_shape, requires_grad);
}


bool Tensor::has_creator() { return !creators.empty(); }

void Tensor::build_grad() {}
