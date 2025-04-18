#include "functional.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>

Tensor add(const Tensor& a, const Tensor& b) {
    // TODO: implement forward and backward logic
    throw std::runtime_error("add not implemented yet");
}

Tensor mul(const Tensor& a, const Tensor& b) {
    assert(!a.shape.empty() && b.shape.empty());
    assert(a.shape.back() == b.shape.front());
    
    std::vector<int> new_shape;
}

Tensor dot(const Tensor& a, const Tensor& b) {
    // TODO: implement forward and backward logic
    throw std::runtime_error("dot not implemented yet");
}

Tensor sigmoid(const Tensor& a) {
    // TODO: implement forward and backward logic
    throw std::runtime_error("sigmoid not implemented yet");
}

Tensor exp(const Tensor& a) {
    // TODO: implement forward and backward logic
    throw std::runtime_error("exp not implemented yet");
}

Tensor log(const Tensor& a) {
    // TODO: implement forward and backward logic
    throw std::runtime_error("log not implemented yet");
}
