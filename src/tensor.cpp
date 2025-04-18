#include "tensor.hpp"
#include <cassert>
#include <iostream>
#include <sstream>

Tensor::Tensor(
    const std::vector<float>& data_,
    const std::vector<int>& shape_,
    bool requires_grad
) : data(data_), shape(shape_), requires_grad(requires_grad) {
    assert(numel_shape(shape) == data.size());

    // Setup strides
    strides = std::vector<int>(shape.size(), 1);
    for(int i = shape.size() - 2; i >= 0; i--) {
        strides.at(i) = strides.at(i+1) * shape.at(i+1);
    }

    // Setup grad
    if (requires_grad) {
        grad = std::vector<float>(data.size(), 0.0f);
    }
}


const std::vector<int>& Tensor::get_shape() const {
    return shape;
}

const std::vector<float>& Tensor::get_data() const {
    return data;
}

void Tensor::set_data(const std::vector<float>& new_data) {
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

float& Tensor::at(std::span<const int> indices) {
    assert(shape.size() == indices.size());

    int flat_index = 0;
    for(int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0 && indices[i] < shape.at(i));
        flat_index += indices[i] * strides.at(i);
    }

    return data.at(flat_index);
}

const float& Tensor::at(std::span<const int> indices) const {
    assert(shape.size() == indices.size());

    int flat_index = 0;
    for(int i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0 && indices[i] < shape.at(i));
        flat_index += indices[i] * strides.at(i);
    }

    return data.at(flat_index);
}

bool Tensor::is_contiguous() const {
    int stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {  
        if (strides.at(i) != stride) return false;
        stride *= shape.at(i);
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
    size_t preview_size = std::min(data.size(), size_t(5));
    for (size_t i = 0; i < preview_size; ++i) {
        ss << data[i];
        if (i != preview_size - 1) {
            ss << ", ";
        }
    }

    if (data.size() > 5) {
        ss << ", ...";
    }

    ss << "])";
    
    return ss.str();
}


bool Tensor::has_creator() { return !creators.empty(); }

void Tensor::build_grad() {}
