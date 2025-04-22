#include "ops/unary_ops.hpp"
#include <omp.h>

template<typename Op>
t_tensor apply_unary_t(const t_tensor& input, Op op) {
    t_data output_data(input->get_data().size());
    const float* __restrict__ data_in = input->get_data().data();
    float* __restrict__ data_out = output_data.data();
    
    #pragma omp parallel for simd
    for(size_t i = 0; i < output_data.size(); i++) {
        data_out[i] = op(data_in[i]);
    }

    return create_tensor(
        output_data,
        t_shape(input->get_shape()),
        input->get_requires_grad()
    );
}

t_tensor apply_unary(const t_tensor& input, std::function<float(float)> op) {
    return apply_unary_t(input, op);
}

template<typename ForwardOp, typename BackwardOp>
t_tensor unary_with_backward(const t_tensor& input, ForwardOp forward_op, BackwardOp backward_op) {
    t_tensor result = apply_unary_t(input, forward_op);

    if (!input->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [input_ptr = input, result_ptr = result, backward_op]() mutable {
        const t_data& grad_output = result_ptr->get_grad();
        t_data& grad_input = input_ptr->get_grad();
        const t_data& data_input = input_ptr->get_data();
        const t_data& data_result = result_ptr->get_data();

        #pragma omp parallel for simd
        for (size_t i = 0; i < grad_input.size(); ++i) {
            grad_input[i] += backward_op(grad_output[i], data_input[i], data_result[i]);
        }
    };

    result->set_requires_grad(true);
    result->set_leaf(false);
    result->set_backward_fn(backward_fn);
    result->add_creator(input);

    return result;
}

t_tensor sigmoid(const t_tensor& input) {
    return unary_with_backward(
        input,
        [](float x) { return sigmoid(x); },
        [](float grad_out, float /*x*/, float y) { return grad_out * y * (1 - y); }
    );
}

t_tensor exp(const t_tensor& input) {
    return unary_with_backward(
        input,
        [](float x) { return exp(x); },
        [](float grad_out, float /*x*/, float y) { return grad_out * y; }
    );
}

t_tensor log(const t_tensor& input) {
    return unary_with_backward(
        input,
        [](float x) { return log(x); },
        [](float grad_out, float x, float /*y*/) { return grad_out / x; }
    );
}
