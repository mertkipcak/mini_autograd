
#include "utils.hpp"
#include <cassert>

int numel_shape(const std::vector<int>& shape) {
    int size = 1;
    for(size_t i = 0; i < shape.size(); i++) {
        size *= shape.at(i);
    }
    return size;
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

