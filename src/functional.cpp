#include "functional.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>


bool same_shape(const Tensor& t1, const Tensor& t2) {
    return t1.get_shape() == t2.get_shape();
}

Tensor add(const Tensor& a, const Tensor& b) {
    assert(same_shape(a, b));
}

Tensor mul(const Tensor& a, const Tensor& b) {
    
}

Tensor dot(const Tensor& a, const Tensor& b) {
    assert(!a.get_shape().empty() && !b.get_shape().empty());
    assert(a.get_shape().back() == b.get_shape().front());
    
    std::vector<int> new_shape = std::vector<int>();
    for(size_t i = 0; i < a.get_shape().size() - 1; i++) 
        new_shape.push_back(a.get_shape()[i]);
    for(size_t i = 1; i < b.get_shape().size(); i++)
        new_shape.push_back(b.get_shape()[i]);

    int new_size = numel_shape(new_shape);
    std::vector<float> new_data(new_size);
    
    bool required_grad = a.get_requires_grad() || b.get_requires_grad();

    Tensor result = Tensor(new_data, new_shape, required_grad);

    // TODO: populate result

    return result;
}

Tensor sigmoid(const Tensor& a) {
    std::vector<float> data(a.get_data());

    for(size_t i = 0; i < data.size(); i++) {
        data[i] = sigmoid(data[i]);
    }

    return Tensor(
        data,
        std::vector<int>(a.get_shape()),
        a.get_requires_grad()
    );
}

Tensor exp(const Tensor& a) {
    std::vector<float> data(a.get_data());

    for(size_t i = 0; i < data.size(); i++) {
        data[i] = exp(data[i]);
    }

    return Tensor(
        data,
        std::vector<int>(a.get_shape()),
        a.get_requires_grad()
    );
}

Tensor log(const Tensor& a) {
    std::vector<float> data(a.get_data());

    for(size_t i = 0; i < data.size(); i++) {
        data[i] = log(data[i]);
    }

    return Tensor(
        data,
        std::vector<int>(a.get_shape()),
        a.get_requires_grad()
    );
}
