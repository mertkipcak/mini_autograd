
#pragma once

#include "utils.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <span>
#include <string>


class Tensor {
public:
    Tensor(const std::vector<float>& data,
           const std::vector<int>& shape,
           bool requires_grad = false);

    const std::vector<int>& get_shape() const;

    const std::vector<float>& get_data() const;
    void set_data(const std::vector<float>& new_data);

    bool get_requires_grad() const;
    void set_requires_grad(bool new_requires_grad);

    float& at(std::span<const int> indices);
    const float& at(std::span<const int> indices) const;

    bool is_contiguous() const;

    void backward();
    void zero_grad();
    size_t numel() const;
    std::string to_string() const;

private:
    std::vector<float> data;
    std::vector<float> grad;
    const std::vector<int> shape;
    std::vector<int> strides;
    bool requires_grad;
    std::vector<std::shared_ptr<Tensor>> creators;
    std::function<void(const Tensor& output_grad)> backward_fn;

    bool has_creator();
    void build_grad();
};
