#include "tensor_iterator.hpp"

TensorIterator::TensorIterator(const t_shape& shape_) : shape(shape_), offset(0), indices(shape_.size(), 0), numel(numel_shape(shape_)) {};

void TensorIterator::inc() {
    offset++;
}

const t_indices TensorIterator::get() {
    size_t remaining = offset;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = static_cast<int>(remaining % shape[i]);
        remaining /= shape[i];
    }
    return t_indices(indices);
}

bool TensorIterator::done() const {
    return offset >= numel;
}
