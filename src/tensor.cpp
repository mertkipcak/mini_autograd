#include "tensor.hpp"
#include <cassert>
#include <iostream>

Tensor::Tensor(
    const std::vector<float>& data_,
    const std::vector<int>& shape_,
    bool requires_grad
) : data(data_), shape(shape_), requires_grad(requires_grad) {
    size_t shape_numel = 1;
    for(size_t i = 0; i < shape.size(); i++) shape_numel *= shape[i];
    assert(shape_numel == data.size());
    data.shrink_to_fit();
    if (requires_grad) {
        grad = std::vector<float>(data.size(), 0.0f);
        grad.shrink_to_fit();
    }
}

float& Tensor::operator[](size_t index) {
    return data.at(index);
}

const float& Tensor::operator[](size_t index) const {
    return data.at(index);
}

float& Tensor::at(size_t index) {
    return data.at(index);
}

const float& Tensor::at(size_t index) const {
    return data.at(index);
}

void Tensor::backward() {}

void Tensor::zero_grad() {
    
}

size_t Tensor::numel() const {
    return data.size();
}

std::string Tensor::to_string() const {
    return "Stub!";
}

bool Tensor::has_creator() { return !creators.empty(); }

void Tensor::build_grad() {}
