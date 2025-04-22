#include "ops/tensor_utils.hpp"
#include "utils.hpp"
#include <omp.h>

t_tensor randn(const t_shape& shape, bool requires_grad) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t numel = numel_shape(shape);
    t_data data(numel);

    for (size_t i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }

    return create_tensor(data, shape, requires_grad);
}

t_tensor sum(const t_tensor& input, int dim, bool keepdims) {
    if (dim < -1 || dim >= static_cast<int>(input->get_shape().size())) {
        throw std::runtime_error("Can only sum across the dim that satisfy -1 <= dim < input.shape().size()");
    }

    t_data data_in = input->get_data();
    const float* __restrict__ in = data_in.data();

    if (dim == -1) {
        float data = 0;

        #pragma omp parallel for
        for(size_t i = 0; i < input->get_data().size(); i++) data += in[i];

        return create_tensor(t_data({data}), t_shape({1, 1}), input->get_requires_grad());
    }

    t_shape shape_out(input->get_shape());
    size_t sum_length = shape_out[dim];
    size_t sum_stride = input->get_strides()[dim];
    if (keepdims) {
        shape_out[dim] = 1;
    } else {
        shape_out.erase(shape_out.begin() + dim);
    }
    size_t final_numel = numel_shape(shape_out);
    t_data data_output(final_numel, 0);
    float* __restrict__ out = data_output.data();

    for(size_t i = 0; i < final_numel; i++) {
        for(size_t j = 0; j < sum_length; j++) {
            out[i] += in[i + sum_stride * j];
        }
    }

    return create_tensor(data_output, shape_out, input->get_requires_grad());
}
