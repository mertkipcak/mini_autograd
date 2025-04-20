
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
class Tensor {
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
               bool make_contiguous = false,
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

        Tensor transpose() const;
    
    private:
        t_data data;         // Tensor values
        t_data grad;         // Gradient values
        const t_shape shape;    // Tensor dimensions
        t_shape strides;        // Memory layout for indexing
        bool contiguous;    // Is memory layed out row-major order
        bool requires_grad;              // Whether to track gradients
        std::vector<std::shared_ptr<Tensor>> creators;  // Input tensors that created this
        std::function<void(const Tensor& output_grad)> backward_fn;  // Function to backpropagate gradients
    
        /**
         * @brief Check if the tensor data is stored in a contiguous layout
         * @return True if the tensor data is contiguous in memory
         */
        bool is_contiguous() const;

        /**
         * @brief Check if the tensor has input creators
         * @return True if the tensor was created by operations on other tensors
         */
        bool has_creator();
        
        /**
         * @brief Initialize gradient storage
         */
        void build_grad();
    };
