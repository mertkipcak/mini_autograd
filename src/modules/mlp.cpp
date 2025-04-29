#include "module.hpp"
#include <cassert>
#include "ops.hpp"

MLP::MLP(const t_shape& layers) {
    assert(layers.size() > 1);
    for(size_t i = 0; i < layers.size() - 1; i++) {
        assert(layers[i] > 0 && layers[i+1] > 0);
        weights.push_back(randn({layers[i+1], layers[i]}, true));
        biases.push_back(randn({layers[i+1]}, true));
    }
}

t_tensor MLP::forward(const t_tensor& input) {
    assert(input->get_shape().size() > 0);
    assert(input->get_shape()[0] == weights[0]->get_shape()[1]);
    t_tensor curr = input;
    for(size_t i = 0; i < weights.size(); i++) {
        curr = matmul(weights[i], curr) + biases[i];
        if (i + 1 < weights.size()) {
            curr = sigmoid(curr);
        }
    }
    return curr;
}

std::vector<t_tensor> MLP::parameters() {
    std::vector<t_tensor> params;
    params.reserve(weights.size() + biases.size());
    params.insert(params.end(), weights.begin(), weights.end());
    params.insert(params.end(), biases.begin(), biases.end());
    return params;
}