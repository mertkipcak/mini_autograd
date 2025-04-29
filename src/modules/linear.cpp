#include "module.hpp"
#include <cassert>
#include "ops.hpp"

Linear::Linear(size_t input_size, size_t output_size, bool use_bias) {
    weight = randn(t_shape({output_size, input_size}), true);
    if (bias) {
        bias = randn(t_shape({output_size}), true);
        this->use_bias = use_bias;
    } else {
        this->use_bias = false;
    }
}

t_tensor Linear::forward(const t_tensor& input) {
    t_tensor result = matmul(weight, input);
    if (use_bias) result = result + bias;
    return result;
}

std::vector<t_tensor> Linear::parameters() {
    std::vector<t_tensor> params;
    params.push_back(weight);
    if (use_bias) params.push_back(bias);
    return params;
}