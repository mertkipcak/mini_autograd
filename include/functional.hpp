#pragma once

#include "tensor.hpp"

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor dot(const Tensor& a, const Tensor& b);
Tensor sigmoid(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
