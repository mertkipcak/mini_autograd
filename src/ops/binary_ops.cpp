#include "ops/binary_ops.hpp"
#include "utils.hpp"
#include <omp.h>

template<typename Op>
t_tensor apply_binary_t(const t_tensor& a, const t_tensor& b, Op op) {
    std::optional<t_shape> maybe_shape = broadcast_shape(a->get_shape(), b->get_shape());
    if (!maybe_shape.has_value()) {
        throw std::runtime_error("Shape mismatch at t_tensor binary operations");
    }

    t_shape res_shape = maybe_shape.value();
    t_shape shape_a = pad_shape_to_size(a->get_shape(), res_shape.size());
    t_shape shape_b = pad_shape_to_size(b->get_shape(), res_shape.size());
    size_t numel = numel_shape(res_shape);
    t_tensor result = create_tensor(t_data(numel), res_shape, a->get_requires_grad() || b->get_requires_grad());

    const float* __restrict__ data_a = a->get_data().data();
    const float* __restrict__ data_b = b->get_data().data();
    float* __restrict__ data_out = result->get_data().data();

    // No broadcasting, both matrices contiguous, optimized path
    if (shape_a == shape_b && a->get_contiguous() && b->get_contiguous()) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < numel; ++i) {
            data_out[i] = op(data_a[i], data_b[i]);
        }
        return result;
    }

    // Broadcasting and/or non-contiguous
    t_shape strides_a = pad_shape_to_size(a->get_strides(), res_shape.size(), 0);
    t_shape strides_b = pad_shape_to_size(b->get_strides(), res_shape.size(), 0);
    
    for (size_t i = 0; i < res_shape.size(); i++) {
        if(shape_a[i] == 1) {
            strides_a[i] = 0;
        }
        if(shape_b[i] == 1) {
            strides_b[i] = 0;
        }
    }
    std::vector<size_t> indices_a(numel);
    std::vector<size_t> indices_b(numel);
    
    #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t remaining = i;
        size_t flat_index_a = 0;
        size_t flat_index_b = 0;
        
        for (size_t dim = res_shape.size() - 1; dim >= 0; --dim) {
            size_t coord = remaining % res_shape[dim];
            flat_index_a += strides_a[dim] * coord;
            flat_index_b += strides_b[dim] * coord;
            remaining /= res_shape[dim];
        }
        
        indices_a[i] = flat_index_a;
        indices_b[i] = flat_index_b;
    }
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < numel; ++i) {
        data_out[i] = op(data_a[indices_a[i]], data_b[indices_b[i]]);
    }
    
    return result;
}

template<typename ForwardOp, typename BackwardOpA, typename BackwardOpB>
t_tensor binary_with_backward(
    const t_tensor& A,
    const t_tensor& B,
    ForwardOp forward_op,
    BackwardOpA backward_op_A,
    BackwardOpB backward_op_B
) {
    t_tensor result = apply_binary_t(A, B, forward_op);

    if (!A->get_requires_grad() && !B->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [A_ptr = A, B_ptr = B, result_ptr = result, backward_op_A, backward_op_B]() mutable {
        const t_data& grad_output = result_ptr->get_grad();
        const t_data& data_A = A_ptr->get_data();
        const t_data& data_B = B_ptr->get_data();
        const t_data& data_result = result_ptr->get_data();
        const t_shape& res_shape = result_ptr->get_shape();
        const t_shape& a_shape = A_ptr->get_shape();
        const t_shape& b_shape = B_ptr->get_shape();
        
        size_t numel = grad_output.size();

        // Broadcasting and/or non-contiguous
        t_shape strides_a = pad_shape_to_size(A_ptr->get_strides(), res_shape.size(), 0);
        t_shape strides_b = pad_shape_to_size(B_ptr->get_strides(), res_shape.size(), 0);
        
        for (size_t i = 0; i < res_shape.size(); i++) {
            if(a_shape.size() <= i || a_shape[i] == 1) {
                strides_a[i] = 0;
            }
            if(b_shape.size() <= i || b_shape[i] == 1) {
                strides_b[i] = 0;
            }
        }
        
        std::vector<size_t> indices_a(numel);
        std::vector<size_t> indices_b(numel);
        
        #pragma omp parallel for
        for (size_t i = 0; i < numel; ++i) {
            size_t remaining = i;
            size_t flat_index_a = 0;
            size_t flat_index_b = 0;
            
            for (size_t dim = res_shape.size() - 1; dim >= 0; --dim) {
                size_t coord = remaining % res_shape[dim];
                flat_index_a += strides_a[dim] * coord;
                flat_index_b += strides_b[dim] * coord;
                remaining /= res_shape[dim];
            }
            
            indices_a[i] = flat_index_a;
            indices_b[i] = flat_index_b;
        }

        if (A_ptr->get_requires_grad()) {
            t_data& grad_A = A_ptr->get_grad();
            
            #pragma omp parallel for
            for (size_t i = 0; i < numel; ++i) {
                float grad_a_val = backward_op_A(grad_output[i], data_A[indices_a[i]], data_B[indices_b[i]], data_result[i]);
                
                #pragma omp atomic
                grad_A[indices_a[i]] += grad_a_val;
            }
        }
        
        if (B_ptr->get_requires_grad()) {
            t_data& grad_B = B_ptr->get_grad();
            
            #pragma omp parallel for
            for (size_t i = 0; i < numel; ++i) {
                float grad_b_val = backward_op_B(grad_output[i], data_A[indices_a[i]], data_B[indices_b[i]], data_result[i]);
                
                #pragma omp atomic
                grad_B[indices_b[i]] += grad_b_val;
            }
        }
    };

    result->set_requires_grad(true);
    result->set_leaf(false);
    result->set_backward_fn(backward_fn);
    
    if (A->get_requires_grad()) {
        result->add_creator(A);
    }
    
    if (B->get_requires_grad()) {
        result->add_creator(B);
    }

    return result;
}

t_tensor apply_binary(const t_tensor& a, const t_tensor& b, std::function<float(float, float)> op) {
    return apply_binary_t(a, b, op);
}

t_tensor add(const t_tensor& a, const t_tensor& b) {
    return binary_with_backward(
        a,
        b,
        [](float a, float b) { return a + b; },
        [](float grad_out, float /*a*/, float /*b*/, float /*out*/) { return grad_out; },
        [](float grad_out, float /*a*/, float /*b*/, float /*out*/) { return grad_out; }
    );
}

t_tensor mul(const t_tensor& a, const t_tensor& b) {
    return binary_with_backward(
        a,
        b,
        [](float a, float b) { return a * b; },
        [](float grad_out, float /*a*/, float b, float /*out*/) { return grad_out * b; },
        [](float grad_out, float a, float /*b*/, float /*out*/) { return grad_out * a; }
    );
}
