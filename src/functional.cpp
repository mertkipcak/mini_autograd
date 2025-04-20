#include "functional.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <omp.h>

Tensor randn(const t_shape& shape, bool requires_grad) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t numel = numel_shape(shape);
    t_data data(numel);

    for (size_t i = 0; i < numel; ++i) data[i] = dist(gen);

    return Tensor(data, shape, requires_grad);
}

bool same_shape(const Tensor& A, const Tensor& B) {
    return A.get_shape() == B.get_shape();
}

std::optional<t_shape> broadcast_shape(const Tensor& A, const Tensor& B) {
    // Setup
    size_t dim1 = A.get_shape().size();
    size_t dim2 = B.get_shape().size();
    size_t max_dim = dim1 > dim2 ? dim1 : dim2;
    t_shape s1 = t_shape(max_dim, 1);
    t_shape s2 = t_shape(max_dim, 1);
    t_shape broadcasted_shape(max_dim);

    // Apply broadcast
    for(size_t i = max_dim; i-- > 0;) {
        if (i + dim1 >= max_dim)
            s1[i] = A.get_shape()[i - max_dim + dim1];
        
        if (i + dim2 >= max_dim)
            s2[i] = B.get_shape()[i - max_dim + dim2];

        if (s1[i] == s2[i]) {
            broadcasted_shape[i] = s1[i];
            continue;
        } else if (s1[i] == 1) {
            broadcasted_shape[i] = s2[i];
            continue;
        } else if (s2[i] == 1) {
            broadcasted_shape[i] = s1[i];
            continue;
        }
        return std::nullopt;
    }

    return broadcasted_shape;
}

template<typename Op>
Tensor apply_binary_t(const Tensor& A, const Tensor& B, Op op) {
    std::optional<t_shape> maybe_shape = broadcast_shape(A, B);
    if (!maybe_shape.has_value()) throw std::runtime_error("Shape mismatch at Tensor binary operations");

    t_shape res_shape = maybe_shape.value();
    t_shape a_shape = pad_shape_to_size(A.get_shape(), res_shape.size());
    t_shape b_shape = pad_shape_to_size(B.get_shape(), res_shape.size());
    size_t numel = numel_shape(res_shape);
    Tensor res(t_data(numel), res_shape, A.get_requires_grad() || B.get_requires_grad());

    const float* __restrict__ a_data = A.get_data().data();
    const float* __restrict__ b_data = B.get_data().data();
    float* __restrict__ out = res.get_data().data();

    // No broadcasting, both matrices contiguous, optimized path
    if (a_shape == b_shape && A.get_contiguous() && B.get_contiguous()) {
        #pragma omp parallel for simd
        for (size_t i = 0; i < numel; ++i) {
            out[i] = op(a_data[i], b_data[i]);
        }
        return res;
    }

    // Broadcasting and/or non-contiguous
    t_shape a_strides = pad_shape_to_size(A.get_strides(), res_shape.size(), 0);
    t_shape b_strides = pad_shape_to_size(B.get_strides(), res_shape.size(), 0);
    
    for (size_t i = 0; i < res_shape.size(); i++) {
        if(a_shape[i] == 1) {
            a_strides[i] = 0;
        }
        if(b_shape[i] == 1) {
            b_strides[i] = 0;
        }
    }
    std::vector<size_t> a_indices(numel);
    std::vector<size_t> b_indices(numel);
    
    #pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t remaining = i;
        size_t a_flat_index = 0;
        size_t b_flat_index = 0;
        
        for (int dim = static_cast<int>(res_shape.size()) - 1; dim >= 0; --dim) {
            size_t coord = remaining % res_shape[dim];
            a_flat_index += a_strides[dim] * coord;
            b_flat_index += b_strides[dim] * coord;
            remaining /= res_shape[dim];
        }
        
        a_indices[i] = a_flat_index;
        b_indices[i] = b_flat_index;
    }
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < numel; ++i) {
        out[i] = op(a_data[a_indices[i]], b_data[b_indices[i]]);
    }
    
    return res;
}

template<typename Op>
Tensor apply_unary_t(const Tensor& A, Op op) {
    t_data data(A.get_data().size());
    const float* __restrict__ in = A.get_data().data();
    float* __restrict__ out = data.data();
    
    #pragma omp parallel for simd
    for(size_t i = 0; i < data.size(); i++) {
        out[i] = op(in[i]);
    }

    return Tensor(
        data,
        std::vector<int>(A.get_shape()),
        A.get_requires_grad()
    );
}

Tensor matmul_contiguous(const Tensor& A, const Tensor& B) {
    const t_shape& shape_A = A.get_shape();
    const t_shape& shape_B = B.get_shape();
    const t_data& data_A = A.get_data();
    const t_data& data_B = B.get_data();

    const int ndim_A = shape_A.size();
    const int M = shape_A[ndim_A - 2];
    const int K = shape_A[ndim_A - 1];
    const int N = shape_B[1];

    const int B1_size = std::accumulate(shape_A.begin(), shape_A.end() - 2, 1, std::multiplies<>());
    const int B2_size = std::accumulate(shape_B.begin() + 2, shape_B.end(), 1, std::multiplies<>());

    // Output shape: [B1..., M, N, B2...]
    t_shape out_shape;
    out_shape.insert(out_shape.end(), shape_A.begin(), shape_A.end() - 2);
    out_shape.push_back(M);
    out_shape.push_back(N);
    out_shape.insert(out_shape.end(), shape_B.begin() + 2, shape_B.end());

    const int total_blocks = B1_size * B2_size;
    const int result_stride = M * N;

    t_data result(total_blocks * result_stride, 0.0f);

    // Block sizes — tuned for L1 cache
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;

    #pragma omp parallel for collapse(2)
    for (int b1 = 0; b1 < B1_size; ++b1) {
        for (int b2 = 0; b2 < B2_size; ++b2) {
            const float* __restrict__ A_batch = data_A.data() + b1 * M * K;
            const float* __restrict__ B_batch = data_B.data() + b2 * K * N;
            float* __restrict__ C_batch = result.data() + (b1 * B2_size + b2) * M * N;

            for (int i0 = 0; i0 < M; i0 += BM) {
                for (int j0 = 0; j0 < N; j0 += BN) {
                    float local_c[BM][BN] = {{0}};

                    const int i_end = std::min(i0 + BM, M);
                    const int j_end = std::min(j0 + BN, N);

                    for (int k0 = 0; k0 < K; k0 += BK) {
                        const int k_end = std::min(k0 + BK, K);
                        for (int i = i0; i < i_end; ++i) {
                            for (int k = k0; k < k_end; ++k) {
                                float a_val = A_batch[i * K + k];
                                #pragma omp simd
                                for (int j = j0; j < j_end; ++j) {
                                    local_c[i - i0][j - j0] += a_val * B_batch[k * N + j];
                                }
                            }
                        }
                    }

                    for (int i = i0; i < i_end; ++i) {
                        for (int j = j0; j < j_end; ++j) {
                            C_batch[i * N + j] = local_c[i - i0][j - j0];
                        }
                    }
                }
            }
        }
    }

    return Tensor(result, out_shape);
}


Tensor matmul_generic(const Tensor& A, const Tensor& B) {
    const t_shape& shape_A = A.get_shape();
    const t_shape& strides_A = A.get_strides();
    const t_shape& shape_B = B.get_shape();
    const t_shape& strides_B = B.get_strides();
    const t_data& data_A = A.get_data();
    const t_data& data_B = B.get_data();

    const int ndim_A = shape_A.size();
    const int M = shape_A[ndim_A - 2];
    const int K = shape_A[ndim_A - 1];
    const int N = shape_B[1];

    t_shape B1_shape(shape_A.begin(), shape_A.end() - 2);
    t_shape B2_shape(shape_B.begin() + 2, shape_B.end());

    // Output shape = [B1..., M, N, B2...]
    t_shape out_shape;
    out_shape.insert(out_shape.end(), B1_shape.begin(), B1_shape.end());
    out_shape.push_back(M);
    out_shape.push_back(N);
    out_shape.insert(out_shape.end(), B2_shape.begin(), B2_shape.end());

    // Compute row-major strides for output
    t_shape out_strides(out_shape.size());
    int stride = 1;
    for (int i = (int)out_shape.size() - 1; i >= 0; --i) {
        out_strides[i] = stride;
        stride *= out_shape[i];
    }

    t_data out_data(stride, 0.0f); // Preallocate

    // Helpers for multi-index traversal
    auto offset = [](const t_shape& index, const t_shape& strides) -> int {
        int off = 0;
        for (size_t i = 0; i < index.size(); ++i) {
            off += index[i] * strides[i];
        }
        return off;
    };

    auto advance_index = [](t_shape& idx, const t_shape& shape) -> bool {
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            if (++idx[i] < shape[i]) return true;
            idx[i] = 0;
        }
        return false;
    };

    // Iterate B1 x B2 index combinations
    t_shape B1_idx(B1_shape.size(), 0);
    t_shape B2_idx(B2_shape.size(), 0);

    do {
        do {
            int offset_A_base = offset(B1_idx, strides_A);
            int offset_B_base = offset(B2_idx, t_shape(strides_B.begin() + 2, strides_B.end()));

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        int a_idx = offset_A_base + i * strides_A[ndim_A - 2] + k * strides_A[ndim_A - 1];
                        int b_idx = k * strides_B[0] + j * strides_B[1] + offset_B_base;
                        acc += data_A[a_idx] * data_B[b_idx];
                    }

                    t_shape out_idx;
                    out_idx.insert(out_idx.end(), B1_idx.begin(), B1_idx.end());
                    out_idx.push_back(i);
                    out_idx.push_back(j);
                    out_idx.insert(out_idx.end(), B2_idx.begin(), B2_idx.end());
                    int out_offset = offset(out_idx, out_strides);
                    out_data[out_offset] = acc;
                }
            }

        } while (advance_index(B2_idx, B2_shape));
    } while (advance_index(B1_idx, B1_shape));

    return Tensor(out_data, out_shape, out_strides);
}

Tensor apply_binary(const Tensor& A, const Tensor& B, std::function<float(float, float)> op) {
    return apply_binary_t(A, B, op);
}

Tensor apply_unary(const Tensor& A, std::function<float(float)> op) {
    return apply_unary_t(A, op);
}

Tensor matmul(const Tensor& A, const Tensor& B) {
    // Assertions
    assert(!A.get_shape().empty() && !B.get_shape().empty());
    assert(A.get_shape().back() == B.get_shape().front());

    // Matmul
    if (A.get_contiguous() && B.get_contiguous()) {
        return matmul_contiguous(A, B);
    } else {
        return matmul_generic(A, B);
    }
}

Tensor add(const Tensor& A, const Tensor& B) {
    return apply_binary_t(A, B, [](float x, float y) { return x + y;});
}

Tensor mul(const Tensor& A, const Tensor& B) {
    return apply_binary_t(A, B, [](float x, float y) { return x * y;});
}

Tensor sigmoid(const Tensor& A) {
    return apply_unary_t(A, [](float x) { return sigmoid(x);});
}

Tensor exp(const Tensor& A) {
    return apply_unary_t(A, [](float x) { return exp(x);});
}

Tensor log(const Tensor& A) {
    return apply_unary_t(A, [](float x) { return log(x);});
}
