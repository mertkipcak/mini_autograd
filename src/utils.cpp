
#include "utils.hpp"
#include <cassert>

int numel_shape(const std::vector<int>& shape) {
    int size = 1;
    for(size_t i = 0; i < shape.size(); i++) {
        size *= shape.at(i);
    }
    return size;
}

t_shape pad_shape_to_size(const t_shape& shape, size_t size, int pad) {
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

