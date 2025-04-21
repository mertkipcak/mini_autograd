#include "functional.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <omp.h>

t_tensor randn(const t_shape& shape, bool requires_grad) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t numel = numel_shape(shape);
    t_data data(numel);

    for (size_t i = 0; i < numel; ++i) {
        data[i] = dist(gen);
    }

    return create_tensor(data, shape, requires_grad);
}

bool same_shape(const t_tensor& a, const t_tensor& b) {
    return a->get_shape() == b->get_shape();
}

std::optional<t_shape> broadcast_shape(const t_tensor& a, const t_tensor& b) {
    // Setup
    size_t dim_a = a->get_shape().size();
    size_t dim_b = b->get_shape().size();
    size_t max_dim = dim_a > dim_b ? dim_a : dim_b;
    t_shape shape_a = t_shape(max_dim, 1);
    t_shape shape_b = t_shape(max_dim, 1);
    t_shape broadcasted_shape(max_dim);

    // Apply broadcast
    for(size_t i = max_dim; i-- > 0;) {
        if (i + dim_a >= max_dim)
            shape_a[i] = a->get_shape()[i - max_dim + dim_a];
        
        if (i + dim_b >= max_dim)
            shape_b[i] = b->get_shape()[i - max_dim + dim_b];

        if (shape_a[i] == shape_b[i]) {
            broadcasted_shape[i] = shape_a[i];
            continue;
        } else if (shape_a[i] == 1) {
            broadcasted_shape[i] = shape_b[i];
            continue;
        } else if (shape_b[i] == 1) {
            broadcasted_shape[i] = shape_a[i];
            continue;
        }
        return std::nullopt;
    }

    return broadcasted_shape;
}

template<typename Op>
t_tensor apply_binary_t(const t_tensor& a, const t_tensor& b, Op op) {
    std::optional<t_shape> maybe_shape = broadcast_shape(a, b);
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
        
        for (int dim = static_cast<int>(res_shape.size()) - 1; dim >= 0; --dim) {
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
        std::vector<int>(input->get_shape()),
        input->get_requires_grad()
    );
}

t_tensor matmul_contiguous(const t_tensor& a, const t_tensor& b) {
    const t_shape& shape_a = a->get_shape();
    const t_shape& shape_b = b->get_shape();
    const t_data& data_a = a->get_data();
    const t_data& data_b = b->get_data();

    const int ndim_a = shape_a.size();
    const int M = shape_a[ndim_a - 2];
    const int K = shape_a[ndim_a - 1];
    const int N = shape_b[1];

    const int batch_size_a = std::accumulate(shape_a.begin(), shape_a.end() - 2, 1, std::multiplies<>());
    const int batch_size_b = std::accumulate(shape_b.begin() + 2, shape_b.end(), 1, std::multiplies<>());

    // Output shape: [batch_a->.., M, N, batch_b->..]
    t_shape out_shape;
    out_shape.insert(out_shape.end(), shape_a.begin(), shape_a.end() - 2);
    out_shape.push_back(M);
    out_shape.push_back(N);
    out_shape.insert(out_shape.end(), shape_b.begin() + 2, shape_b.end());

    const int total_batches = batch_size_a * batch_size_b;
    const int result_stride = M * N;

    t_data result_data(total_batches * result_stride, 0.0f);

    // Block sizes — tuned for L1 cache
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;

    #pragma omp parallel for collapse(2)
    for (int batch_a = 0; batch_a < batch_size_a; ++batch_a) {
        for (int batch_b = 0; batch_b < batch_size_b; ++batch_b) {
            const float* __restrict__ batch_data_a = data_a.data() + batch_a * M * K;
            const float* __restrict__ batch_data_b = data_b.data() + batch_b * K * N;
            float* __restrict__ batch_data_out = result_data.data() + (batch_a * batch_size_b + batch_b) * M * N;

            for (int i0 = 0; i0 < M; i0 += BM) {
                for (int j0 = 0; j0 < N; j0 += BN) {
                    float local_c[BM][BN] = {{0}};

                    const int i_end = std::min(i0 + BM, M);
                    const int j_end = std::min(j0 + BN, N);

                    for (int k0 = 0; k0 < K; k0 += BK) {
                        const int k_end = std::min(k0 + BK, K);
                        for (int i = i0; i < i_end; ++i) {
                            for (int k = k0; k < k_end; ++k) {
                                float val_a = batch_data_a[i * K + k];
                                #pragma omp simd
                                for (int j = j0; j < j_end; ++j) {
                                    local_c[i - i0][j - j0] += val_a * batch_data_b[k * N + j];
                                }
                            }
                        }
                    }

                    for (int i = i0; i < i_end; ++i) {
                        for (int j = j0; j < j_end; ++j) {
                            batch_data_out[i * N + j] = local_c[i - i0][j - j0];
                        }
                    }
                }
            }
        }
    }

    return create_tensor(result_data, out_shape);
}


t_tensor matmul_generic(const t_tensor& a, const t_tensor& b) {
    const t_shape& shape_a = a->get_shape();
    const t_shape& strides_a = a->get_strides();
    const t_shape& shape_b = b->get_shape();
    const t_shape& strides_b = b->get_strides();
    const t_data& data_a = a->get_data();
    const t_data& data_b = b->get_data();

    const int ndim_a = shape_a.size();
    const int M = shape_a[ndim_a - 2];
    const int K = shape_a[ndim_a - 1];
    const int N = shape_b[1];

    t_shape batch_shape_a(shape_a.begin(), shape_a.end() - 2);
    t_shape batch_shape_b(shape_b.begin() + 2, shape_b.end());

    // Output shape = [batch_a->.., M, N, batch_b->..]
    t_shape out_shape;
    out_shape.insert(out_shape.end(), batch_shape_a.begin(), batch_shape_a.end());
    out_shape.push_back(M);
    out_shape.push_back(N);
    out_shape.insert(out_shape.end(), batch_shape_b.begin(), batch_shape_b.end());

    // Compute row-major strides for output
    t_shape out_strides(out_shape.size());
    int stride = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= out_shape[i];
    }

    t_data out_data(stride, 0.0f); // Preallocate

    // Helpers for multi-index traversal
    auto calculate_offset = [](const t_shape& index, const t_shape& strides) -> int {
        int offset = 0;
        for (size_t i = 0; i < index.size(); ++i) {
            offset += index[i] * strides[i];
        }
        return offset;
    };

    auto advance_index = [](t_shape& index, const t_shape& shape) -> bool {
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            if (++index[i] < shape[i]) return true;
            index[i] = 0;
        }
        return false;
    };

    // Iterate batch_a x batch_b index combinations
    t_shape batch_index_a(batch_shape_a.size(), 0);
    t_shape batch_index_b(batch_shape_b.size(), 0);

    do {
        do {
            int offset_a_base = calculate_offset(batch_index_a, strides_a);
            int offset_b_base = calculate_offset(batch_index_b, t_shape(strides_b.begin() + 2, strides_b.end()));

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        int index_a = offset_a_base + i * strides_a[ndim_a - 2] + k * strides_a[ndim_a - 1];
                        int index_b = k * strides_b[0] + j * strides_b[1] + offset_b_base;
                        acc += data_a[index_a] * data_b[index_b];
                    }

                    t_shape out_index;
                    out_index.insert(out_index.end(), batch_index_a.begin(), batch_index_a.end());
                    out_index.push_back(i);
                    out_index.push_back(j);
                    out_index.insert(out_index.end(), batch_index_b.begin(), batch_index_b.end());
                    int out_offset = calculate_offset(out_index, out_strides);
                    out_data[out_offset] = acc;
                }
            }

        } while (advance_index(batch_index_b, batch_shape_b));
    } while (advance_index(batch_index_a, batch_shape_a));

    return create_tensor(out_data, out_shape, out_strides);
}

t_tensor apply_binary(const t_tensor& a, const t_tensor& b, std::function<float(float, float)> op) {
    return apply_binary_t(a, b, op);
}

t_tensor apply_unary(const t_tensor& input, std::function<float(float)> op) {
    return apply_unary_t(input, op);
}

t_tensor matmul(const t_tensor& a, const t_tensor& b) {
    // Assertions
    assert(!a->get_shape().empty() && !b->get_shape().empty());
    assert(a->get_shape().back() == b->get_shape().front());

    // Matmul
    if (a->get_contiguous() && b->get_contiguous()) {
        return matmul_contiguous(a, b);
    } else {
        return matmul_generic(a, b);
    }
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
            
            for (int dim = static_cast<int>(res_shape.size()) - 1; dim >= 0; --dim) {
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

t_tensor add(const t_tensor& a, const t_tensor& b) {
    return binary_with_backward(
        a,
        b,
        [](float a, float b) { return a + b; },
        [](float grad_out, float a, float b, float out) { return grad_out; },
        [](float grad_out, float a, float b, float out) { return grad_out; }
    );
}

t_tensor mul(const t_tensor& a, const t_tensor& b) {
    return binary_with_backward(
        a,
        b,
        [](float a, float b) { return a * b; },
        [](float grad_out, float a, float b, float out) { return grad_out * b; },
        [](float grad_out, float a, float b, float out) { return grad_out * a; }
    );
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

t_tensor sum(const t_tensor& input, int dim) {
    if (dim < -1 || dim >= input->get_shape().size()) {
        throw std::runtime_error("Can only sum across the dim that satisfy -1 <= dim < input.shape().size()");
    }

    t_data data_input = input->get_data();

    if (dim == -1) {
        const float* __restrict__ data_vec = data_input.data();
        float data = 0;

        #pragma omp parallel for
        for(size_t i = 0; i < input->get_data().size(); i++) data += data_vec[i];

        return create_tensor(t_data({data}), t_shape({1, 1}), input->get_requires_grad());
    }   
}
