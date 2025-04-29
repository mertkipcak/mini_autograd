#pragma once

#include "tensor.hpp"
#include "ops.hpp"
#include <memory>
#include <vector>

class Module;

using t_module = std::shared_ptr<Module>;
using t_params = std::vector<t_tensor>;

/**
 * @brief A neural network module interface
 */
class Module : public std::enable_shared_from_this<Module> {
    public:
        virtual ~Module() = default;

        virtual t_tensor forward(const t_tensor& input) = 0;
        virtual std::vector<t_tensor> parameters() = 0;
        
        void zero_grad() {
            for(auto& param : parameters()) {
                param->zero_grad();
            }
        };

        t_tensor operator()(const t_tensor& input) { return forward(input); };
};

/**
 * @brief Helper template to create modules with proper shared_ptr management
 */
template<typename T, typename... Args>
inline t_module create_module(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

class Linear : public Module {
    public:
        Linear(size_t input_size, size_t output_size, bool use_bias = true);

        t_tensor forward(const t_tensor& input) override;
        std::vector<t_tensor> parameters() override;

    private:
        t_tensor weight;
        t_tensor bias;
        bool use_bias;
};

class ReLU : public Module {
    public:
        ReLU() = default;

        t_tensor forward(const t_tensor& input) override {
            return relu(input);
        };
        
        std::vector<t_tensor> parameters() override {
            return {};
        };
};

class Sequential : public Module {
    public:
        Sequential(std::vector<t_module> modules_) : modules(std::move(modules_)) {}
        
        template<typename... Modules>
        Sequential(Modules... args) {
            (modules.push_back(args), ...);
        }

        t_tensor forward(const t_tensor& input) override {
            t_tensor curr = input;
            for(const auto& module : modules) {
                curr = module->forward(curr);
            }
            return curr;
        };
        
        std::vector<t_tensor> parameters() override {
            std::vector<t_tensor> params;
            for(const auto& module : modules) {
                auto module_params = module->parameters();
                params.insert(params.end(), module_params.begin(), module_params.end());
            }
            return params;
        }

    private:
        std::vector<t_module> modules;
};

class MLP : public Module {
    public:
        MLP(const t_shape& layers);

        t_tensor forward(const t_tensor& input) override;
        std::vector<t_tensor> parameters() override;

    private:
        t_params weights;
        t_params biases;
};