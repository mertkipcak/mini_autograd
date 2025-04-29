
#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>

/**
 * @brief A neural network module interface
 */
class Module : public std::enable_shared_from_this<Module> {
    public:
        virtual ~Module() = default;

        virtual t_tensor forward(const t_tensor& input) = 0;
        virtual std::vector<t_tensor> parameters() = 0;
        void zero_grad() {
            for(t_tensor param : parameters()) {
                param->zero_grad();
            }
        };
};

using t_params = std::vector<t_tensor>;
using t_module = std::shared_ptr<Module>;
template<typename... Args>
inline t_module create_module(Args&&... args) {
    return std::make_shared<Module>(std::forward<Args>(args)...);
}

class MLP : public Module {
    public:
        MLP(const t_shape& layers);

        t_tensor forward(const t_tensor& input) override;
        std::vector<t_tensor> parameters() override;

    private:
        t_params matmul_params;
        t_params bias_params;
};