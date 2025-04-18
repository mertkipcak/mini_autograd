
#pragma once

#include "utils.hpp"
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
        Tensor(const std::vector<float>& data,
               const std::vector<int>& shape,
               bool requires_grad = false);
    
        /**
         * @brief Get the tensor's shape
         * @return A vector of dimensions
         */
        const std::vector<int>& get_shape() const;
    
        /**
         * @brief Get the tensor's data
         * @return A flattened vector of values
         */
        const std::vector<float>& get_data() const;
        
        /**
         * @brief Set the tensor's data
         * @param new_data The new data values as a flattened vector
         */
        void set_data(const std::vector<float>& new_data);
    
        /**
         * @brief Check if the tensor requires gradient computation
         * @return True if gradients are being tracked
         */
        bool get_requires_grad() const;
        
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
        float& at(std::span<const int> indices);
        
        /**
         * @brief Const access to a tensor element by multi-dimensional indices
         * @param indices A span of indices for each dimension
         * @return Const reference to the value at the specified position
         */
        const float& at(std::span<const int> indices) const;
    
        /**
         * @brief Check if the tensor data is stored in a contiguous layout
         * @return True if the tensor data is contiguous in memory
         */
        bool is_contiguous() const;
    
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
    
    private:
        std::vector<float> data;         // Tensor values
        std::vector<float> grad;         // Gradient values
        const std::vector<int> shape;    // Tensor dimensions
        std::vector<int> strides;        // Memory layout for indexing
        bool requires_grad;              // Whether to track gradients
        std::vector<std::shared_ptr<Tensor>> creators;  // Input tensors that created this
        std::function<void(const Tensor& output_grad)> backward_fn;  // Function to backpropagate gradients
    
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
