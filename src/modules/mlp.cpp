#include "module.hpp"
#include <cassert>
#include "ops.hpp"

MLP::MLP(const t_shape& layers) {
    assert(layers.size() > 1);
    for(size_t i = 0; i < layers.size() - 1; i++) {
        assert(layers[i] > 0 && layers[i+1] > 0);
        matmul_params.push_back(randn({layers[i+1], layers[i]}, true));
        bias_params.push_back(randn({layers[i+1]}, true));
    }
}

t_tensor MLP::forward(const t_tensor& input) {
    assert(input->get_shape().size() > 0);
    assert(input->get_shape()[0] == matmul_params[0]->get_shape()[1]);
    t_tensor curr = input;
    for(size_t i = 0; i < matmul_params.size(); i++) {
        curr = matmul(matmul_params[i], curr) + bias_params[i];
        if (i + 1 < matmul_params.size()) {
            curr = sigmoid(curr);
        }
    }
    return curr;
}

std::vector<t_tensor> MLP::parameters() {
    std::vector<t_tensor> params;
    params.reserve(matmul_params.size() + bias_params.size());
    params.insert(params.end(), matmul_params.begin(), matmul_params.end());
    params.insert(params.end(), bias_params.begin(), bias_params.end());
    return params;
}