#pragma once

#include "types.hpp"
#include "utils.hpp"

class TensorIterator {
    public:
        TensorIterator(const t_shape& shape_);

        void inc();
        const t_indices get();
        bool done() const;

    private:
        t_shape shape;
        size_t offset;
        t_shape indices;
        size_t numel;
};