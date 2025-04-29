#include "ops.hpp"
#include <cassert>
#include <numeric>
#include <omp.h>

t_tensor matmul(const t_tensor& a, const t_tensor& b) {
    // Validate inputs
    const t_shape& shape_a = a->get_shape();
    const t_shape& shape_b = b->get_shape();
    const t_shape& strides_a = a->get_strides();
    const t_shape& strides_b = b->get_strides();
    const t_data& data_a = a->get_data();
    const t_data& data_b = b->get_data();
    
    // Ensure valid shapes for matmul
    assert(!shape_a.empty() && !shape_b.empty());
    
    // Handle 1D vector case for B
    bool b_is_1d = shape_b.size() == 1;
    
    // For 1D vector, we treat it as a column vector with shape [K, 1]
    // For matrix multiplication: A[..., M, K] @ B[K] -> C[..., M]
    if (b_is_1d) {
        assert(shape_a.back() == shape_b[0]); // Last dim of A must match only dim of B
    } else {
        assert(shape_a.back() == shape_b.front()); // Last dim of A must match first dim of B
    }
    
    // Get dimensions for matrix multiplication
    const size_t ndim_a = shape_a.size();
    const size_t M = shape_a[ndim_a - 2];  // Second-last dimension of A
    const size_t K = shape_a[ndim_a - 1];  // Last dimension of A
    const size_t N = b_is_1d ? 1 : shape_b[1];  // For 1D vector, N=1
    
    // Extract batch dimensions
    t_shape batch_shape_a(shape_a.begin(), shape_a.end() - 2);
    t_shape batch_shape_b;
    if (!b_is_1d) {
        batch_shape_b.insert(batch_shape_b.end(), shape_b.begin() + 2, shape_b.end());
    }
    
    // Calculate batch sizes
    const size_t batch_size_a = std::accumulate(batch_shape_a.begin(), batch_shape_a.end(), 
                                               1, std::multiplies<>());
    const size_t batch_size_b = std::accumulate(batch_shape_b.begin(), batch_shape_b.end(), 
                                               1, std::multiplies<>());
    
    // Construct output shape
    t_shape out_shape;
    out_shape.insert(out_shape.end(), batch_shape_a.begin(), batch_shape_a.end());
    
    if (b_is_1d) {
        // For 1D vectors, output is [batch_a..., M]
        out_shape.push_back(M);
    } else {
        // For 2D+ matrices, output is [batch_a..., M, N, batch_b...]
        out_shape.push_back(M);
        out_shape.push_back(N);
        out_shape.insert(out_shape.end(), batch_shape_b.begin(), batch_shape_b.end());
    }
    
    // Compute row-major strides for output
    t_shape out_strides(out_shape.size());
    size_t stride = 1;
    for (int i = out_shape.size() - 1; i >= 0; i--) {
        out_strides[i] = stride;
        stride *= out_shape[i];
    }
    
    // Calculate total output size
    const size_t total_elements = std::accumulate(out_shape.begin(), out_shape.end(), 
                                                 1, std::multiplies<>());
    
    // Preallocate output data
    t_data out_data(total_elements, 0.0f);
    
    // Perform the matrix multiplication
    #pragma omp parallel for collapse(2)
    for (size_t batch_idx_a = 0; batch_idx_a < batch_size_a; batch_idx_a++) {
        for (size_t batch_idx_b = 0; batch_idx_b < batch_size_b; batch_idx_b++) {
            // Convert flat batch indices to multi-dimensional indices
            t_shape batch_indices_a(batch_shape_a.size());
            size_t remaining_a = batch_idx_a;
            for (int i = batch_shape_a.size() - 1; i >= 0; i--) {
                batch_indices_a[i] = remaining_a % batch_shape_a[i];
                remaining_a /= batch_shape_a[i];
            }
            
            t_shape batch_indices_b(batch_shape_b.size());
            size_t remaining_b = batch_idx_b;
            for (int i = batch_shape_b.size() - 1; i >= 0; i--) {
                batch_indices_b[i] = remaining_b % batch_shape_b[i];
                remaining_b /= batch_shape_b[i];
            }
            
            // Calculate base offsets for these batch indices
            size_t offset_a_base = 0;
            for (size_t i = 0; i < batch_indices_a.size(); i++) {
                offset_a_base += batch_indices_a[i] * strides_a[i];
            }
            
            size_t offset_b_base = 0;
            if (!b_is_1d) {
                for (size_t i = 0; i < batch_indices_b.size(); i++) {
                    offset_b_base += batch_indices_b[i] * strides_b[i + 2];
                }
            }
            
            // Calculate base output offset
            size_t out_offset_base = 0;
            for (size_t i = 0; i < batch_indices_a.size(); i++) {
                out_offset_base += batch_indices_a[i] * out_strides[i];
            }
            
            size_t matrix_idx_offset = batch_shape_a.size();
            if (!b_is_1d) {
                size_t batch_b_offset = matrix_idx_offset + 2;
                for (size_t i = 0; i < batch_indices_b.size(); i++) {
                    out_offset_base += batch_indices_b[i] * out_strides[batch_b_offset + i];
                }
            }
            
            // Matrix multiplication for this batch combination
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    float acc = 0.0f;
                    // Dot product along K dimension
                    for (size_t k = 0; k < K; k++) {
                        size_t index_a = offset_a_base + i * strides_a[ndim_a - 2] + k * strides_a[ndim_a - 1];
                        size_t index_b;
                        if (b_is_1d) {
                            index_b = k * strides_b[0];  // B is a 1D vector
                        } else {
                            index_b = k * strides_b[0] + j * strides_b[1] + offset_b_base;
                        }
                        acc += data_a[index_a] * data_b[index_b];
                    }
                    
                    // Store result
                    size_t out_idx;
                    if (b_is_1d) {
                        out_idx = out_offset_base + i * out_strides[matrix_idx_offset];
                    } else {
                        out_idx = out_offset_base + i * out_strides[matrix_idx_offset] + 
                                j * out_strides[matrix_idx_offset + 1];
                    }
                    out_data[out_idx] = acc;
                }
            }
        }
    }
    
    // Create result tensor
    t_tensor result = create_tensor(out_data, out_shape, 
                                   a->get_requires_grad() || b->get_requires_grad());
    
    // Add backward function if needed
    if (a->get_requires_grad() || b->get_requires_grad()) {
        std::function<void()> backward_fn = [
            a_weak = std::weak_ptr<Tensor>(a),
            b_weak = std::weak_ptr<Tensor>(b),
            result_weak = std::weak_ptr<Tensor>(result),
            shape_a, strides_a, ndim_a,
            shape_b, strides_b,
            out_shape, out_strides,
            batch_shape_a, batch_shape_b,
            M, N, K, b_is_1d
        ]() {
            auto a_ptr = a_weak.lock();
            auto b_ptr = b_weak.lock();
            auto result_ptr = result_weak.lock();
            if (!a_ptr || !b_ptr || !result_ptr) return;
            
            const t_data& grad_output = result_ptr->get_grad();
            
            // Calculate batch sizes
            size_t batch_size_a = std::accumulate(batch_shape_a.begin(), batch_shape_a.end(), 
                                                 1, std::multiplies<>());
            size_t batch_size_b = std::accumulate(batch_shape_b.begin(), batch_shape_b.end(), 
                                                 1, std::multiplies<>());
            
            // Backprop to A if needed
            if (a_ptr->get_requires_grad()) {
                t_data& grad_a = a_ptr->get_grad();
                
                #pragma omp parallel for collapse(2)
                for (size_t batch_idx_a = 0; batch_idx_a < batch_size_a; batch_idx_a++) {
                    for (size_t batch_idx_b = 0; batch_idx_b < batch_size_b; batch_idx_b++) {
                        // Convert flat batch indices to multi-dimensional indices
                        t_shape batch_indices_a(batch_shape_a.size());
                        size_t remaining_a = batch_idx_a;
                        for (int i = batch_shape_a.size() - 1; i >= 0; i--) {
                            batch_indices_a[i] = remaining_a % batch_shape_a[i];
                            remaining_a /= batch_shape_a[i];
                        }
                        
                        t_shape batch_indices_b(batch_shape_b.size());
                        size_t remaining_b = batch_idx_b;
                        for (int i = batch_shape_b.size() - 1; i >= 0; i--) {
                            batch_indices_b[i] = remaining_b % batch_shape_b[i];
                            remaining_b /= batch_shape_b[i];
                        }
                        
                        // Calculate base offsets
                        size_t offset_a_base = 0;
                        for (size_t i = 0; i < batch_indices_a.size(); i++) {
                            offset_a_base += batch_indices_a[i] * strides_a[i];
                        }
                        
                        size_t offset_b_base = 0;
                        if (!b_is_1d) {
                            for (size_t i = 0; i < batch_indices_b.size(); i++) {
                                offset_b_base += batch_indices_b[i] * strides_b[i + 2];
                            }
                        }
                        
                        // Calculate base output offset
                        size_t out_offset_base = 0;
                        for (size_t i = 0; i < batch_indices_a.size(); i++) {
                            out_offset_base += batch_indices_a[i] * out_strides[i];
                        }
                        
                        size_t matrix_idx_offset = batch_shape_a.size();
                        if (!b_is_1d) {
                            size_t batch_b_offset = matrix_idx_offset + 2;
                            for (size_t i = 0; i < batch_indices_b.size(); i++) {
                                out_offset_base += batch_indices_b[i] * out_strides[batch_b_offset + i];
                            }
                        }
                        
                        // For each element in A, compute its gradient
                        for (size_t i = 0; i < M; i++) {
                            for (size_t k = 0; k < K; k++) {
                                float grad_sum = 0.0f;
                                
                                if (b_is_1d) {
                                    // For 1D vector: dA[i,k] = dC[i] * B[k]
                                    size_t out_idx = out_offset_base + i * out_strides[matrix_idx_offset];
                                    size_t b_idx = k * strides_b[0];
                                    grad_sum = grad_output[out_idx] * b_ptr->get_data()[b_idx];
                                } else {
                                    // For matrix: dA[i,k] = sum_j(dC[i,j] * B[k,j])
                                    for (size_t j = 0; j < N; j++) {
                                        size_t out_idx = out_offset_base + i * out_strides[matrix_idx_offset] + 
                                                        j * out_strides[matrix_idx_offset + 1];
                                        size_t b_idx = k * strides_b[0] + j * strides_b[1] + offset_b_base;
                                        
                                        grad_sum += grad_output[out_idx] * b_ptr->get_data()[b_idx];
                                    }
                                }
                                
                                // Update gradient for A
                                size_t a_idx = offset_a_base + i * strides_a[ndim_a - 2] + k * strides_a[ndim_a - 1];
                                #pragma omp atomic
                                grad_a[a_idx] += grad_sum;
                            }
                        }
                    }
                }
            }
            
            // Backprop to B if needed
            if (b_ptr->get_requires_grad()) {
                t_data& grad_b = b_ptr->get_grad();
                
                #pragma omp parallel for collapse(2)
                for (size_t batch_idx_a = 0; batch_idx_a < batch_size_a; batch_idx_a++) {
                    for (size_t batch_idx_b = 0; batch_idx_b < batch_size_b; batch_idx_b++) {
                        // Convert flat batch indices to multi-dimensional indices
                        t_shape batch_indices_a(batch_shape_a.size());
                        size_t remaining_a = batch_idx_a;
                        for (int i = batch_shape_a.size() - 1; i >= 0; i--) {
                            batch_indices_a[i] = remaining_a % batch_shape_a[i];
                            remaining_a /= batch_shape_a[i];
                        }
                        
                        t_shape batch_indices_b(batch_shape_b.size());
                        size_t remaining_b = batch_idx_b;
                        for (int i = batch_shape_b.size() - 1; i >= 0; i--) {
                            batch_indices_b[i] = remaining_b % batch_shape_b[i];
                            remaining_b /= batch_shape_b[i];
                        }
                        
                        // Calculate base offsets
                        size_t offset_a_base = 0;
                        for (size_t i = 0; i < batch_indices_a.size(); i++) {
                            offset_a_base += batch_indices_a[i] * strides_a[i];
                        }
                        
                        size_t offset_b_base = 0;
                        if (!b_is_1d) {
                            for (size_t i = 0; i < batch_indices_b.size(); i++) {
                                offset_b_base += batch_indices_b[i] * strides_b[i + 2];
                            }
                        }
                        
                        // Calculate base output offset
                        size_t out_offset_base = 0;
                        for (size_t i = 0; i < batch_indices_a.size(); i++) {
                            out_offset_base += batch_indices_a[i] * out_strides[i];
                        }
                        
                        size_t matrix_idx_offset = batch_shape_a.size();
                        if (!b_is_1d) {
                            size_t batch_b_offset = matrix_idx_offset + 2;
                            for (size_t i = 0; i < batch_indices_b.size(); i++) {
                                out_offset_base += batch_indices_b[i] * out_strides[batch_b_offset + i];
                            }
                        }
                        
                        if (b_is_1d) {
                            // For 1D vector: dB[k] = sum_i(A[i,k] * dC[i])
                            for (size_t k = 0; k < K; k++) {
                                float grad_sum = 0.0f;
                                for (size_t i = 0; i < M; i++) {
                                    size_t out_idx = out_offset_base + i * out_strides[matrix_idx_offset];
                                    size_t a_idx = offset_a_base + i * strides_a[ndim_a - 2] + k * strides_a[ndim_a - 1];
                                    
                                    grad_sum += grad_output[out_idx] * a_ptr->get_data()[a_idx];
                                }
                                
                                // Update gradient for B
                                size_t b_idx = k * strides_b[0];
                                #pragma omp atomic
                                grad_b[b_idx] += grad_sum;
                            }
                        } else {
                            // For matrix: dB[k,j] = sum_i(A[i,k] * dC[i,j])
                            for (size_t k = 0; k < K; k++) {
                                for (size_t j = 0; j < N; j++) {
                                    float grad_sum = 0.0f;
                                    // Sum over M dimension
                                    for (size_t i = 0; i < M; i++) {
                                        size_t out_idx = out_offset_base + i * out_strides[matrix_idx_offset] + 
                                                        j * out_strides[matrix_idx_offset + 1];
                                        size_t a_idx = offset_a_base + i * strides_a[ndim_a - 2] + k * strides_a[ndim_a - 1];
                                        
                                        grad_sum += grad_output[out_idx] * a_ptr->get_data()[a_idx];
                                    }
                                    
                                    // Update gradient for B
                                    size_t b_idx = k * strides_b[0] + j * strides_b[1] + offset_b_base;
                                    #pragma omp atomic
                                    grad_b[b_idx] += grad_sum;
                                }
                            }
                        }
                    }
                }
            }
        };
        
        result->set_backward_fn(backward_fn);
        if (a->get_requires_grad()) {
            result->add_creator(a);
        }
        if (b->get_requires_grad()) {
            result->add_creator(b);
        }
    }
    
    return result;
}