#include "ops.hpp"
#include "utils.hpp"

t_tensor randn(const t_shape& shape, bool requires_grad) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    size_t numel = numel_shape(shape);
    t_data data(numel);

    for (size_t i = 0; i < numel; i++) {
        data[i] = dist(gen);
    }

    return create_tensor(data, shape, requires_grad);
}

t_tensor zeros(const t_shape& shape, bool requires_grad) {
    size_t numel = numel_shape(shape);
    t_data data(numel, 0);

    return create_tensor(data, shape, requires_grad);
}

t_tensor sumall(const t_tensor& input) {
    t_data data_in = input->get_data();
    const float* __restrict__ in = data_in.data();
    float data = 0;

    #pragma omp parallel for reduction(+:data)
    for(size_t i = 0; i < input->get_data().size(); i++) data += in[i];

    t_tensor result = create_tensor(t_data({data}), t_shape({}), input->get_requires_grad());

    if (!result->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [input, result]() mutable {
        const t_data& grad_output = result->get_grad();
        t_data& grad_input = input->get_grad();

        #pragma omp parallel for
        for (size_t i = 0; i < grad_input.size(); i++) {
            grad_input[i] += grad_output[0];
        }
    };

    result->set_backward_fn(backward_fn);
    result->add_creator(input);

    return result;
}

t_tensor sum(const t_tensor& input, size_t dim, bool keepdims) {
    if (dim >= input->get_shape().size()) {
        throw std::runtime_error("Can only sum across the dim that satisfy 0 <= dim < input.shape().size()");
    }

    const t_shape& input_shape = input->get_shape();
    const t_shape& input_strides = input->get_strides();
    const float* __restrict__ in_data = input->get_data().data();
    
    t_shape output_shape = input_shape;
    if (keepdims) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
    }
    
    size_t output_size = numel_shape(output_shape);
    t_data output_data(output_size, 0.0f);
    float* __restrict__ out_data = output_data.data();
    
    size_t dim_size = input_shape[dim];
    size_t dim_stride = input_strides[dim];
    
    size_t outer_size = 1;
    for (size_t i = 0; i < dim; i++) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); i++) {
        inner_size *= input_shape[i];
    }
    
    #pragma omp parallel for collapse(2)
    for (size_t outer = 0; outer < outer_size; outer++) {
        for (size_t inner = 0; inner < inner_size; inner++) {
            size_t output_idx = outer * inner_size + inner;
            
            size_t base_input_idx = 0;
            size_t remaining_outer = outer;
            for (size_t i = 0; i < dim; i++) {
                size_t dim_idx = remaining_outer % input_shape[i];
                remaining_outer /= input_shape[i];
                base_input_idx += dim_idx * input_strides[i];
            }
            
            size_t remaining_inner = inner;
            for (size_t i = dim + 1; i < input_shape.size(); i++) {
                size_t dim_idx = remaining_inner % input_shape[i];
                remaining_inner /= input_shape[i];
                base_input_idx += dim_idx * input_strides[i];
            }
            
            float sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (size_t d = 0; d < dim_size; d++) {
                sum += in_data[base_input_idx + d * dim_stride];
            }
            out_data[output_idx] = sum;
        }
    }
    
    t_tensor result =  create_tensor(output_data, output_shape, input->get_requires_grad());

    if (!result->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [
        input,
        result,
        dim,
        input_shape,
        input_strides,
        dim_size,
        dim_stride,
        outer_size,
        inner_size
    ]() {
        const t_data& grad_output = result->get_grad();
        t_data& grad_input = input->get_grad();
        
        #pragma omp parallel for collapse(2)
        for (size_t outer = 0; outer < outer_size; outer++) {
            for (size_t inner = 0; inner < inner_size; inner++) {
                size_t output_idx = outer * inner_size + inner;
                float grad_val = grad_output[output_idx];
                
                size_t base_input_idx = 0;
                size_t remaining_outer = outer;
                for (size_t i = 0; i < dim; i++) {
                    size_t dim_idx = remaining_outer % input_shape[i];
                    remaining_outer /= input_shape[i];
                    base_input_idx += dim_idx * input_strides[i];
                }
                
                size_t remaining_inner = inner;
                for (size_t i = dim + 1; i < input_shape.size(); i++) {
                    size_t dim_idx = remaining_inner % input_shape[i];
                    remaining_inner /= input_shape[i];
                    base_input_idx += dim_idx * input_strides[i];
                }
                
                for (size_t d = 0; d < dim_size; d++) {
                    size_t input_idx = base_input_idx + d * dim_stride;
                    #pragma omp atomic
                    grad_input[input_idx] += grad_val;
                }
            }
        }
    };

    result->set_backward_fn(backward_fn);
    result->add_creator(input);

    return result;
}

t_tensor softmax(const t_tensor& input, float temperature) {
    t_data data_out = t_data(input->get_data());
    float* __restrict__ out = data_out.data();
    float max_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < data_out.size(); i++) {
        max_val = std::max(max_val, out[i]);
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < data_out.size(); i++) {
        out[i] -= max_val;
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < data_out.size(); i++) {
        out[i] = std::exp(out[i] / temperature);
    }

    float total = 0;
    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < data_out.size(); i++) {
        total += out[i];
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < data_out.size(); i++) {
        out[i] /= total;
    }

    t_tensor result = create_tensor(data_out, t_shape(input->get_shape()), input->get_requires_grad());

    if (!result->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [input, result]() mutable {
        const t_data& data_output = result->get_data();
        const t_data& grad_output = result->get_grad();
        t_data& grad_input = input->get_grad();
        #pragma omp parallel for
        for (size_t i = 0; i < grad_input.size(); i++) {
            const float grad = grad_output[i] * (data_output[i] * (1.0f - data_output[i]));
            grad_input[i] += grad;
        }
    };

    result->set_backward_fn(backward_fn);
    result->add_creator(input);

    return result;
}

t_tensor cross_entropy(const t_tensor& input, size_t correct_index, float temperature) {
    t_data probs = t_data(input->get_data());
    float* __restrict__ out = probs.data();
    float max_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < probs.size(); i++) {
        max_val = std::max(max_val, out[i]);
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < probs.size(); i++) {
        out[i] -= max_val;
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < probs.size(); i++) {
        out[i] = std::exp(out[i] / temperature);
    }

    float total = 0;
    #pragma omp parallel for reduction(+:total)
    for (size_t i = 0; i < probs.size(); i++) {
        total += out[i];
    }

    #pragma omp parallel for simd
    for (size_t i = 0; i < probs.size(); i++) {
        out[i] /= total;
    }

    t_tensor result = create_tensor(t_data({-std::log(probs[correct_index])}), t_shape({}), input->get_requires_grad());

    if (!result->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [input, result, probs, correct_index]() mutable {
        const t_data& grad_output = result->get_grad();
        t_data& grad_input = input->get_grad();
        #pragma omp parallel for
        for (size_t i = 0; i < grad_input.size(); i++) {
            if (i == correct_index)
                grad_input[i] += grad_output[0] * (probs[i] - 1);
            else
                grad_input[i] += grad_output[0] * probs[i];
        }
    };

    result->set_backward_fn(backward_fn);
    result->add_creator(input);

    return result;
}

t_tensor mean(const t_tensor& input) {
    t_data data_input = input->get_data();
    const float* __restrict__ in = data_input.data();
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for(size_t i = 0; i < data_input.size(); i++) sum += in[i];

    t_tensor result = create_tensor(t_data({sum / static_cast<float>(data_input.size())}), t_shape({}), input->get_requires_grad());

    if (!result->get_requires_grad()) {
        return result;
    }

    std::function<void()> backward_fn = [input, result]() mutable {
        const t_data& grad_output = result->get_grad();
        t_data& grad_input = input->get_grad();

        #pragma omp parallel for
        for (size_t i = 0; i < grad_input.size(); i++) {
            grad_input[i] += grad_output[0] / static_cast<float>(grad_input.size());
        }
    };

    result->set_backward_fn(backward_fn);
    result->add_creator(input);

    return result;
}

t_tensor dropout(const t_tensor& input, float p) {
    if (p <= 0.0f) return input;
    if (p >= 1.0f) return zeros_like(input);
    
    const size_t len = input->get_data().size();
    t_data out_data(len);
    
    const float scale = 1.0f / (1.0f - p);
    const float* __restrict__ in = input->get_data().data();
    float* __restrict__ out = out_data.data();
    
    std::shared_ptr<std::vector<uint8_t>> mask = 
        std::make_shared<std::vector<uint8_t>>(len);
    uint8_t* __restrict__ mask_data = mask->data();
    
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < len; i++) {
        if (dis(gen) < p) {
            mask_data[i] = 0;
            out[i] = 0.0f;
        } else {
            mask_data[i] = 1;
            out[i] = in[i] * scale;
        }
    }
    
    t_tensor result = create_tensor(out_data, input->get_shape(), input->get_requires_grad());
    
    if (result->get_requires_grad()) {
        std::function<void()> backward_fn = [input, result, mask, scale]() {
            const t_data& grad_output = result->get_grad();
            t_data& grad_input = input->get_grad();
            const uint8_t* __restrict__ mask_ptr = mask->data();
            const float* __restrict__ grad_out = grad_output.data();
            float* __restrict__ grad_in = grad_input.data();
            
            #pragma omp parallel for
            for (size_t i = 0; i < grad_input.size(); i++) {
                if (mask_ptr[i]) {
                    grad_in[i] += grad_out[i] * scale;
                }
            }
        };
        
        result->set_backward_fn(backward_fn);
        result->add_creator(input);
    }
    
    return result;
}