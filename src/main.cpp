#include "tensor.hpp"
#include "types.hpp"
#include "ops.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cmath>
#include <random>
#include <iomanip>
#include <functional>

// Template for generating polynomial features for any function
void generate_polynomial_features(t_data& data_X, t_data& data_y, 
                                   size_t n, size_t d, 
                                   std::function<float(float)> target_function,
                                   float min_x = 0.0f, float max_x = 6.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_x, max_x);

    data_X.clear();
    data_y.clear();

    for (size_t i = 0; i < n; ++i) {
        float x = dist(gen);
        float x_power = 1.0f;

        for (size_t p = 0; p < d; ++p) {
            data_X.push_back(x_power);
            x_power *= x;
        }

        data_y.push_back(target_function(x));
    }
}

// Function to generate test data
std::pair<t_data, t_data> generate_test_data(size_t n_test, size_t d, 
                                             std::function<float(float)> target_function,
                                             float min_x = 0.0f, float max_x = 6.0f) {
    t_data test_X, test_y;
    
    // Fixed seed for reproducibility
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min_x, max_x);
    
    for (size_t i = 0; i < n_test; ++i) {
        float x = dist(gen);
        float x_power = 1.0f;
        
        for (size_t p = 0; p < d; ++p) {
            test_X.push_back(x_power);
            x_power *= x;
        }
        
        test_y.push_back(target_function(x));
    }
    
    return {test_X, test_y};
}

void polynomial_regression(std::function<float(float)> target_function, 
                            size_t n = 50, size_t d = 2,
                            float min_x = 0.0f, float max_x = 6.0f,
                            float lr = 1e-2, float lambda_val = 1e-3, int epochs = 1000) {
    // Generate training data
    t_data data_X, data_y;
    generate_polynomial_features(data_X, data_y, n, d, target_function, min_x, max_x);
    
    t_tensor X = create_tensor(data_X, t_shape({n, d}), false);
    t_tensor y = create_tensor(data_y, t_shape({n, 1}), false);
    
    // Initialize weights
    t_tensor w = randn(t_shape({d, 1}), true);
    t_tensor lambda = create_tensor(t_data({lambda_val}), t_shape({}));
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        t_tensor y_pred = matmul(X, w);
        t_tensor accuracy = div(sumall(square(sub(y_pred, y))), create_tensor(t_data({static_cast<float>(n)}), t_shape({})));
        t_tensor regularization = sumall(mul(lambda, square(w)));
        t_tensor loss = add(accuracy, regularization);
        loss->backward();

        // Gradient descent step
        for (size_t i = 0; i < w->get_data().size(); ++i) {
            w->get_data()[i] -= lr * w->get_grad()[i];
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss->get_data()[0] << std::endl;
        }
    }
    
    std::cout << "Training complete. Final weights:\n";
    for (size_t i = 0; i < d; ++i) {
        std::cout << "w[" << i << "]: " << w->get_data()[i] << std::endl;
    }
    
    // Generate test data
    size_t n_test = 10;
    auto [test_X_data, test_y_data] = generate_test_data(n_test, d, target_function, min_x, max_x);
    
    t_tensor test_X = create_tensor(test_X_data, t_shape({n_test, d}), false);
    t_tensor test_y = create_tensor(test_y_data, t_shape({n_test, 1}), false);
    
    // Make predictions
    t_tensor test_pred = matmul(test_X, w);
    
    // Calculate test loss
    t_tensor test_loss = div(sumall(square(sub(test_pred, test_y))),
                              create_tensor(t_data({static_cast<float>(n_test)}), t_shape({})));
    
    std::cout << "\nTest Results for " << n_test << " examples:\n";
    std::cout << "Test MSE: " << test_loss->get_data()[0] << std::endl;
    std::cout << "\nDetailed predictions:\n";
    std::cout << std::setw(15) << "Original x" << std::setw(15) << "True y" 
              << std::setw(15) << "Predicted" << std::setw(15) << "Error" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t i = 0; i < n_test; ++i) {
        float x = test_X_data[i * d + 1]; // x^1 term
        float true_y = test_y_data[i];
        float pred_y = test_pred->get_data()[i];
        float error = std::abs(true_y - pred_y);
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(15) << x << std::setw(15) << true_y 
                  << std::setw(15) << pred_y << std::setw(15) << error << std::endl;
    }
}

int main() {
    std::cout << "\n\nPolynomial regression for 3x + 2:" << std::endl;
    polynomial_regression([](float x) { return 3*x + 2; });
    return 0;
}
