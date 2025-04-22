
#pragma once

#include "utils.hpp"
#include "types.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <span>
#include <string>

/**
 * @brief A multidimensional array implementation supporting automatic differentiation
 */
class Tensor : public std::enable_shared_from_this<Tensor> {
    public:
        /**
         * @brief Construct a new Tensor
         * @param data The tensor's values as a flattened vector
         * @param shape The dimensions of the tensor
         * @param requires_grad Whether to track gradients for this tensor
         */
        Tensor(const t_data& data,
               const t_shape& shape,
               bool requires_grad = false);

        Tensor(const t_data& data,
               const t_shape& shape,
               const t_shape& strides,
               bool requires_grad = false);

        Tensor(Tensor& other,
               bool requires_grad = false);
    
        /**
         * @brief Get the tensor's shape
         * @return A vector of dimensions
         */
        const t_shape& get_shape() const { return shape; };

        /**
         * @brief Get the tensor's strides
         * @return A vector of strides
         */
        const t_shape& get_strides() const { return strides; };
    
        /**
         * @brief Get the tensor's data
         * @return A flattened vector of values
         */
        const t_data& get_data() const { return data; };

        /**
         * @brief Get the tensor's data
         * @return A flattened vector of values
         */
        t_data& get_data() { return data; };
        
        /**
         * @brief Set the tensor's data
         * @param new_data The new data values as a flattened vector
         */
        void set_data(const t_data& new_data) { data = new_data; };

        /**
         * @brief Check if the tensor is stored in row-major order
         * @return True if memory in row-major order
         */
        bool get_contiguous() const { return contiguous; };
    
        /**
         * @brief Check if the tensor requires gradient computation
         * @return True if gradients are being tracked
         */
        bool get_requires_grad() const { return requires_grad; };
        
        /**
         * @brief Set whether the tensor requires gradient computation
         * @param new_requires_grad New gradient tracking setting
         */
        void set_requires_grad(bool new_requires_grad);

        /**
         * @brief Get the tensor's grad
         * @return A flattened vector of values
         */
        t_data& get_grad() { return grad; };
    
        /**
         * @brief Access a tensor element by multi-dimensional indices
         * @param indices A span of indices for each dimension
         * @return Reference to the value at the specified position
         */
        float& at(t_indices& indices);

        /**
         * @brief Const access to a tensor element by multi-dimensional indices
         * @param indices A span of indices for each dimension
         * @return Const reference to the value at the specified position
         */
        const float& at(t_indices& indices) const;

        /**
         * @brief Access the broadcasted value at specific indices
         * @param broadcast_shape The shape to which the tensor is being broadcasted
         * @param indices The multi-dimensional indices in the broadcasted shape
         * @return Const reference to the value at the specified broadcasted position
         */
        const float& broadcasted_at(const t_indices& indices, const t_shape& broadcast_shape) const;

        /**
         * @brief Access a grad element by multi-dimensional indices
         * @param indices A span of indices for each dimension
         * @return Reference to the grad value at the specified position
         */
        float& grad_at(t_indices& indices);

        /**
         * @brief Const access to a grad element by multi-dimensional indices
         * @param indices A span of indices for each dimension
         * @return Const reference to the grad value at the specified position
         */
        const float& grad_at(t_indices& indices) const;

        /**
         * @brief Add a new creator to this matrix
         * @param creator Pointer to the tensor that created this
         * @return Adds creator to the list of creators
         */
        void add_creator(const std::shared_ptr<Tensor> creator) { creators.push_back(creator); }

        /**
         * @brief Get the list of creator tensors for this tensor
         * @return A vector of shared pointers to the creator tensors
         */
        const std::vector<std::shared_ptr<Tensor>> get_creators() const { return creators; }

        /**
         * @brief Check if the tensor is a leaf node in the computation graph
         * @return True if the tensor is a leaf (not the result of an operation)
         */
        bool is_leaf() const { return leaf; };

        /**
         * @brief Set whether the tensor is a leaf node in the computation graph
         * @param leaf_ True if the tensor should be marked as a leaf
         */
        void set_leaf(bool leaf_) { leaf = leaf_; }

        /**
         * @brief Set the function to be run on autodoff
         * @param fn The function to set backward_fn to
         */
        void set_backward_fn(std::function<void()> fn) { backward_fn = fn; }
    
        /**
         * @brief Perform backpropagation starting from this tensor
         */
        void backward();
        
        /**
         * @brief Zero out the gradient values
         */
        void zero_grad();
        
        /**
         * @brief Get total number of elements in the tensor
         * @return Size of the tensor
         */
        size_t numel() const;
        
        /**
         * @brief Get a string representation of the tensor
         * @return Formatted string showing tensor shape and values
         */
        std::string to_string() const;

        std::shared_ptr<Tensor> transpose() const;
    
    private:
        t_data data;         // Tensor values
        t_data grad;         // Gradient values
        t_shape shape;    // Tensor dimensions
        t_shape strides;        // Memory layout for indexing
        bool contiguous;    // Is memory layed out row-major order
        bool requires_grad;              // Whether to track gradients
        bool leaf;
        std::vector<std::shared_ptr<Tensor>> creators;  // Input tensors that created this
        std::function<void()> backward_fn;  // Function to backpropagate gradients
    
        /**
         * @brief Check if the tensor data is stored in a contiguous layout
         * @return True if the tensor data is contiguous in memory
         */
        bool is_contiguous() const;

        /**
         * @brief Check if the tensor has input creators
         * @return True if the tensor was created by operations on other tensors
         */
        bool has_creator() const;
        
        /**
         * @brief Initialize gradient storage
         */
        std::vector<std::shared_ptr<Tensor>> topo_sort();

        /**
         * @brief Get the flat index of the data given indices
         */
        int get_flat_index(const t_indices& indices) const;
};

using t_tensor = std::shared_ptr<Tensor>;
template<typename... Args>
inline t_tensor create_tensor(Args&&... args) {
    return std::make_shared<Tensor>(std::forward<Args>(args)...);
}
