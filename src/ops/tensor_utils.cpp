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

t_tensor sum(const t_tensor& input, int dim) {
    if (dim < -1 || dim >= static_cast<int>(input->get_shape().size())) {
        throw std::runtime_error("Can only sum across the dim that satisfy -1 <= dim < input.shape().size()");
    }

    t_data data_input = input->get_data();

    if (dim == -1) {
        const float* __restrict__ data_vec = data_input.data();
        float data = 0;

        #pragma omp parallel for
        for(size_t i = 0; i < input->get_data().size(); i++) data += data_vec[i];

        return create_tensor(t_data({data}), t_shape({1, 1}), input->get_requires_grad());
    }

    const float* __restrict__ data_vec = data_input.data();
        float data = 0;

        #pragma omp parallel for
        for(size_t i = 0; i < input->get_data().size(); i++) data += data_vec[i];

        return create_tensor(t_data({data}), t_shape({1, 1}), input->get_requires_grad());
}
