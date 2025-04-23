#include "ops.hpp"
#include <cassert>
#include <numeric>
#include <omp.h>

t_tensor matmul_contiguous(const t_tensor& a, const t_tensor& b) {
    const t_shape& shape_a = a->get_shape();
    const t_shape& shape_b = b->get_shape();
    const t_data& data_a = a->get_data();
    const t_data& data_b = b->get_data();

    const size_t ndim_a = shape_a.size();
    const size_t M = shape_a[ndim_a - 2];
    const size_t K = shape_a[ndim_a - 1];
    const size_t N = shape_b[1];

    const size_t batch_size_a = std::accumulate(shape_a.begin(), shape_a.end() - 2, 1, std::multiplies<>());
    const size_t batch_size_b = std::accumulate(shape_b.begin() + 2, shape_b.end(), 1, std::multiplies<>());

    // Output shape: [batch_a->.., M, N, batch_b->..]
    t_shape out_shape;
    out_shape.insert(out_shape.end(), shape_a.begin(), shape_a.end() - 2);
    out_shape.push_back(M);
    out_shape.push_back(N);
    out_shape.insert(out_shape.end(), shape_b.begin() + 2, shape_b.end());

    const size_t total_batches = batch_size_a * batch_size_b;
    const size_t result_stride = M * N;

    t_data result_data(total_batches * result_stride, 0.0f);

    // Block sizes — tuned for L1 cache
    constexpr size_t BM = 64;
    constexpr size_t BN = 64;
    constexpr size_t BK = 64;

    #pragma omp parallel for collapse(2)
    for (size_t batch_a = 0; batch_a < batch_size_a; ++batch_a) {
        for (size_t batch_b = 0; batch_b < batch_size_b; ++batch_b) {
            const float* __restrict__ batch_data_a = data_a.data() + batch_a * M * K;
            const float* __restrict__ batch_data_b = data_b.data() + batch_b * K * N;
            float* __restrict__ batch_data_out = result_data.data() + (batch_a * batch_size_b + batch_b) * M * N;

            for (size_t i0 = 0; i0 < M; i0 += BM) {
                for (size_t j0 = 0; j0 < N; j0 += BN) {
                    float local_c[BM][BN] = {{0}};

                    const size_t i_end = std::min(i0 + BM, M);
                    const size_t j_end = std::min(j0 + BN, N);

                    for (size_t k0 = 0; k0 < K; k0 += BK) {
                        const size_t k_end = std::min(k0 + BK, K);
                        for (size_t i = i0; i < i_end; ++i) {
                            for (size_t k = k0; k < k_end; ++k) {
                                float val_a = batch_data_a[i * K + k];
                                #pragma omp simd
                                for (size_t j = j0; j < j_end; ++j) {
                                    local_c[i - i0][j - j0] += val_a * batch_data_b[k * N + j];
                                }
                            }
                        }
                    }

                    for (size_t i = i0; i < i_end; ++i) {
                        for (size_t j = j0; j < j_end; ++j) {
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

    const size_t ndim_a = shape_a.size();
    const size_t M = shape_a[ndim_a - 2];
    const size_t K = shape_a[ndim_a - 1];
    const size_t N = shape_b[1];

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
    size_t stride = 1;
    for (size_t i = out_shape.size(); i-- > 0;) {
        out_strides[i] = stride;
        stride *= out_shape[i];
    }

    t_data out_data(stride, 0.0f); // Preallocate

    // Helpers for multi-index traversal
    auto calculate_offset = [](const t_shape& index, const t_shape& strides) -> size_t {
        size_t offset = 0;
        for (size_t i = 0; i < index.size(); ++i) {
            offset += index[i] * strides[i];
        }
        return offset;
    };

    auto advance_index = [](t_shape& index, const t_shape& shape) -> bool {
        for (size_t i = shape.size(); i-- > 0;) {
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
            size_t offset_a_base = calculate_offset(batch_index_a, strides_a);
            size_t offset_b_base = calculate_offset(batch_index_b, t_shape(strides_b.begin() + 2, strides_b.end()));

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        size_t index_a = offset_a_base + i * strides_a[ndim_a - 2] + k * strides_a[ndim_a - 1];
                        size_t index_b = k * strides_b[0] + j * strides_b[1] + offset_b_base;
                        acc += data_a[index_a] * data_b[index_b];
                    }

                    t_shape out_index;
                    out_index.insert(out_index.end(), batch_index_a.begin(), batch_index_a.end());
                    out_index.push_back(i);
                    out_index.push_back(j);
                    out_index.insert(out_index.end(), batch_index_b.begin(), batch_index_b.end());
                    size_t out_offset = calculate_offset(out_index, out_strides);
                    out_data[out_offset] = acc;
                }
            }

        } while (advance_index(batch_index_b, batch_shape_b));
    } while (advance_index(batch_index_a, batch_shape_a));

    return create_tensor(out_data, out_shape, out_strides);
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