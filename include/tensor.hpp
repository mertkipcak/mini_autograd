
#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;
    bool requires_grad;
    std::vector<std::shared_ptr<Tensor>> creators;
    std::function<void(const Tensor& output_grad)> backward_fn;

    Tensor(const std::vector<float>& data,
           const std::vector<int>& shape,
           bool requires_grad = false);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& at(size_t index);
    const float& at(size_t index) const;

    void backward();
    void zero_grad();
    size_t numel() const;
    std::string to_string() const;

private:
    bool has_creator();
    void build_grad();
};
