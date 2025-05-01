#pragma once

#include "tensor.hpp"
#include <functional>
#include <optional>
#include <vector>
#include <random>

/*ADD****************************************************************************************************************/
/**
 * @brief Create a new tensor that is the element-wise sum of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor add(const t_tensor& a, const t_tensor& b);

inline t_tensor operator+(const t_tensor& a, const t_tensor& b) {
    return add(a, b);
}

inline t_tensor operator+(const t_tensor& a, float b) {
    return add(a, create_tensor(t_data({b}), t_shape({})));
}

inline t_tensor operator+(float a, const t_tensor& b) {
    return add(create_tensor(t_data({a}), t_shape({})), b);
}
/********************************************************************************************************************/

/*SUB****************************************************************************************************************/
/**
 * @brief Create a new tensor that is the element-wise sub of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result with elements a - b
 */
t_tensor sub(const t_tensor& a, const t_tensor& b);

inline t_tensor operator-(const t_tensor& a, const t_tensor& b) {
    return sub(a, b);
}

inline t_tensor operator-(const t_tensor& a, float b) {
    return sub(a, create_tensor(t_data({b}), t_shape({})));
}

inline t_tensor operator-(float a, const t_tensor& b) {
    return sub(create_tensor(t_data({a}), t_shape({})), b);
}
/********************************************************************************************************************/

/*MUL****************************************************************************************************************/
/**
 * @brief Create a new tensor that is the element-wise product of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor mul(const t_tensor& a, const t_tensor& b);

inline t_tensor operator*(const t_tensor& a, const t_tensor& b) {
    return mul(a, b);
}

inline t_tensor operator*(const t_tensor& a, float b) {
    return mul(a, create_tensor(t_data({b}), t_shape({})));
}

inline t_tensor operator*(float a, const t_tensor& b) {
    return mul(create_tensor(t_data({a}), t_shape({})), b);
}

/********************************************************************************************************************/

/*DIV****************************************************************************************************************/
/**
 * @brief Create a new tensor that is the element-wise division of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor div(const t_tensor& a, const t_tensor& b);

inline t_tensor operator/(const t_tensor& a, const t_tensor& b) {
    return div(a, b);
}

inline t_tensor operator/(const t_tensor& a, float b) {
    return div(a, create_tensor(t_data({b}), t_shape({})));
}

inline t_tensor operator/(float a, const t_tensor& b) {
    return div(create_tensor(t_data({a}), t_shape({})), b);
}

/********************************************************************************************************************/

/**
 * @brief Create a new tensor from the tensor multiplication of two tensors
 * @param a First tensor
 * @param b Second tensor
 * @return New tensor containing the result
 */
t_tensor matmul(const t_tensor& a, const t_tensor& b);

/**
 * @brief Get a tensor with a given shape initialized with normal random weights
 * @param shape Shape of the return tensor
 * @param requires_grad Wheter to track gradients or not
 * @return Random tensor
 */
t_tensor randn(const t_shape& shape, bool requires_grad=false);

/**
 * @brief A vector of all zeros with the shape same as input
 * @param input Input tensor
 * @param requires_grad Wheter to track gradients or not
 * @return Tensor with all zeros
 */
inline t_tensor zeros_like(const t_tensor& input, bool requires_grad = false) {
    return zeros(input->get_shape(), requires_grad);
}

/**
 * @brief A vector of all zeros with the shape same as input
 * @param input Input tensor
 * @return Tensor with all zeros
 */
t_tensor zeros(const t_shape& shape, bool requires_grad=false);

/**
 * @brief sum the elements of the tensor, across a given dimension
 */
t_tensor sum(const t_tensor& input, size_t dim, bool keepdims = false);

/**
 * @brief sum all the elements of the tensor
 */
t_tensor sumall(const t_tensor& input);

/**
 * @brief Converts the tensor into probabilities
 */
t_tensor softmax(const t_tensor& input, float temperature = 1);

/**
 * @brief Gets the cross entropy los from the tensor
 */
t_tensor cross_entropy(const t_tensor& input, size_t correct_index, float temperature = 1);

/**
 * @brief apply sigmoid function element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with sigmoid applied
 */
t_tensor sigmoid(const t_tensor& input);

/**
 * @brief apply exponential function element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with exp applied
 */
t_tensor exp(const t_tensor& input);

/**
 * @brief apply natural logarithm element-wise to a tensor
 * @param input Input tensor
 * @return New tensor with log applied
 */
t_tensor log(const t_tensor& input);

/**
 * @brief square element-wise a tensor
 * @param input Input tensor
 * @return New squared tensor
 */
t_tensor square(const t_tensor& input);

/**
 * @brief relu element-wise a tensor
 * @param input Input tensor
 * @return New relu tensor
 */
t_tensor relu(const t_tensor& input);

/**
 * @brief mean of all the elements
 * @param input Input tensor
 * @return Tensor with mean value
 */
t_tensor mean(const t_tensor& input);

/**
 * @brief sets 0 of all elements with a chance of p
 * @param input Input tensor
 * @return Tensor with dropped values
 */
t_tensor dropout(const t_tensor& input, float prob);