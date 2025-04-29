#include "tensor.hpp"
#include "module.hpp"
#include "ops.hpp"
#include <iostream>
#include <vector>

// Very simple mean squared error loss
t_tensor mse_loss(const t_tensor& pred, const t_tensor& target) {
    t_tensor diff = pred - target;
    return mean(diff * diff);
}

// Manual SGD step
void gd_step(std::vector<t_tensor>& parameters, float lr) {
    for (auto& param : parameters) {
        if (param->get_requires_grad()) {
            t_data& data = param->get_data();
            t_data& grad = param->get_grad();
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] -= lr * grad[i];
            }
        }
    }
}

int main() {
    // XOR dataset
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    std::vector<float> targets = {0.0f, 1.0f, 1.0f, 0.0f};

    t_shape layers = {2, 4, 1};
    t_module model = create_module<MLP>(layers);

    // Optimizer settings
    float lr = 0.1f;
    int epochs = 1000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Create input tensor
            t_tensor x = create_tensor(t_data({inputs[i]}), t_shape({2}), false);
            t_tensor y = create_tensor(t_data({targets[i]}), t_shape({1}), false);

            // Forward pass
            t_tensor output = model->forward(x);

            // Compute loss
            t_tensor loss = mse_loss(output, y);
            total_loss += loss->get_data()[0];

            // Backward pass
            model->zero_grad();
            loss->backward();

            // GD step
            auto params = model->parameters();
            gd_step(params, lr);
        }

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " Loss: " << total_loss / inputs.size() << std::endl;
        }
    }

    // Evaluate model
    std::cout << "\nTrained XOR results:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        t_tensor x = create_tensor(t_data({inputs[i]}), t_shape({2}), false);
        t_tensor output = model->forward(x);
        std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << output->get_data()[0] << std::endl;
    }

    return 0;
}
