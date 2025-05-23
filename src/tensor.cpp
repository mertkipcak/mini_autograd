#include "tensor.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <unordered_set>

Tensor::Tensor(
    const t_data& data_,
    const t_shape& shape_,
    bool requires_grad_
) : data(data_), shape(shape_), requires_grad(requires_grad_) {
    assert(numel_shape(shape) == data.size());

    // Setup strides
    strides = t_shape(shape.size(), 1);
    if (shape.size() > 1) 
        for(size_t i = shape.size() - 1; i-- > 0;)
            strides[i] = strides[i+1] * shape[i+1];

    contiguous = is_contiguous();

    // Setup grad
    if (requires_grad) {
        grad = t_data(data.size(), 0.0f);
    }
}

Tensor::Tensor(
    const t_data& data_,
    const t_shape& shape_,
    const t_shape& strides_,
    bool requires_grad
) : data(data_), shape(shape_), strides(strides_), requires_grad(requires_grad) {
    assert(numel_shape(shape) == data.size());

    contiguous = is_contiguous();

    // Setup grad
    if (requires_grad) {
        grad = t_data(data.size(), 0.0f);
    }
}

Tensor::Tensor(
    Tensor& other,
    bool requires_grad
) : std::enable_shared_from_this<Tensor>(other), data(other.data), shape(other.shape), strides(other.strides), requires_grad(requires_grad) {
    contiguous = is_contiguous();

    // Setup grad
    if (requires_grad) {
        grad = t_data(data.size(), 0.0f);
    }
}

void Tensor::set_requires_grad(bool new_requires_grad) {
    requires_grad = new_requires_grad;

    if (requires_grad && grad.size() != data.size()) {
        grad.resize(data.size(), 0.0f);
    }
}

size_t Tensor::get_flat_index(const t_indices& indices) const {
    size_t flat_index = 0;
    for(size_t i = 0; i < shape.size(); i++) {
        assert(indices[i] >= 0 && indices[i] < shape[i]);
        flat_index += indices[i] * strides[i];
    }

    return flat_index;
}

float& Tensor::at(t_indices& indices) {
    return data.at(get_flat_index(indices));
}

const float& Tensor::at(t_indices& indices) const {
    return data.at(get_flat_index(indices));
}


const float& Tensor::broadcasted_at(const t_indices& indices, const t_shape& broadcasted_shape) const {
    size_t flat_index = 0;
    for(size_t i = 1; i <= shape.size(); i++) {
        size_t bo = broadcasted_shape.size() - i; // broadcasted offset
        size_t o = shape.size() - i; // offset
        assert(broadcasted_shape[bo] == shape[o] || shape[o] == 1);
        if (shape[o] == 1) continue;
        flat_index += indices[bo] * strides[o];
    }

    return data.at(flat_index);
}

float& Tensor::grad_at(t_indices& indices) {
    return grad.at(get_flat_index(indices));
}

const float& Tensor::grad_at(t_indices& indices) const {
    return grad.at(get_flat_index(indices));
};

bool Tensor::is_contiguous() const {
    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {  
        if (strides[i] != stride) return false;
        stride *= shape[i];
    }

    return true;
}

std::string Tensor::to_string() const {
    size_t limit = 50;
    std::stringstream ss;

    ss << "Tensor(";
    ss << "shape=";
    for (size_t i = 0; i < shape.size(); i++) {
        ss << shape[i];
        if (i != shape.size() - 1) {
            ss << "x";
        }
    }

    ss << ", strides=[";
    for (size_t i = 0; i < strides.size(); i++) {
        ss << strides[i];
        if (i != strides.size() - 1) {
            ss << ", ";
        }
    }

    ss << "], is_contiguous=" << (is_contiguous() ? "True" : "False");

    ss << ", requires_grad=" << (requires_grad ? "True" : "False");

    ss << ", data=[";
    size_t preview_size = std::min(data.size(), limit);
    for (size_t i = 0; i < preview_size; i++) {
        ss << data[i];
        if (i != preview_size - 1) {
            ss << ", ";
        }
    }

    if (data.size() > limit) {
        ss << ", ...";
    }

    ss << "])";

    if (!requires_grad) return ss.str();

    ss << ", grad=[";
    for (size_t i = 0; i < preview_size; i++) {
        ss << grad[i];
        if (i != preview_size - 1) {
            ss << ", ";
        }
    }

    if (grad.size() > limit) {
        ss << ", ...";
    }

    ss << "])";
    
    return ss.str();
}

t_tensor Tensor::transpose() const {
    if (shape.size() == 1) {
        t_shape new_shape = {shape[0], 1};
        t_shape new_strides = {1, shape[0]};
        return create_tensor(data, new_shape, new_strides, requires_grad);
    }

    t_shape new_shape(shape);
    std::swap(new_shape[new_shape.size() - 2], new_shape[new_shape.size() - 1]);

    t_shape new_strides(strides);
    std::swap(new_strides[new_strides.size() - 2], new_strides[new_strides.size() - 1]);

    // If tensor is contiguous, we return a view-like result by simply adjusting the shape and strides.
    if (get_contiguous()) {
        return create_tensor(data, new_shape, new_strides, requires_grad);
    }

    // If tensor is not contiguous, we will need to copy the data
    t_data new_data(data.size());
    
    size_t stride_in = strides[strides.size() - 2];
    size_t stride_out = new_strides[strides.size() - 2];
    
    size_t idx_in, idx_out;
    for (size_t i = 0; i < shape[shape.size() - 2]; i++) {
        for (size_t j = 0; j < shape[shape.size() - 1]; j++) {
            idx_in = i * stride_in + j;
            idx_out = j * stride_out + i;
            new_data[idx_out] = data[idx_in];
        }
    }

    // Return a new tensor with transposed data and updated shape and strides
    return create_tensor(new_data, new_shape, new_strides, requires_grad);
}

void Tensor::backward() {
    assert(requires_grad);
    assert(numel() == 1);
    
    std::vector<t_tensor> all_nodes = topo_sort();

    for(const t_tensor& node : all_nodes)
        node->zero_grad();

    grad = t_data({1});

    for(size_t i = all_nodes.size(); i-- > 0;) {
        t_tensor& node = all_nodes[i];
        if (node->get_requires_grad() && !node->creators.empty())
            node->backward_fn();
    }
        
}

void Tensor::zero_grad() {
    grad.assign(data.size(), 0.0f);
}

size_t Tensor::numel() const {
    return data.size();
}

bool Tensor::has_creator() const { return !creators.empty(); }

std::vector<t_tensor> Tensor::topo_sort() {
    std::vector<t_tensor> sorted_tensors;
    std::unordered_set<t_tensor> visited;
    
    std::function<void(const t_tensor&)> dfs = [&](const t_tensor& node) {
        if (visited.contains(node)) return;
        visited.insert(node);
        for (const t_tensor& creator : node->creators) {
            dfs(creator);
        }

        sorted_tensors.push_back(node);
    };
    
    dfs(shared_from_this());
    
    return sorted_tensors;
}
