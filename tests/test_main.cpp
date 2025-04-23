#include "tensor.hpp"
#include "ops.hpp"
#include <cassert>

void test_add() {
    t_tensor A = create_tensor(t_data({1, 2, 3}), t_shape({3}));
    t_tensor B = create_tensor(t_data({4, 5, 6}), t_shape({3}));
    t_tensor C = add(A, B);
    t_data expected({5, 7, 9});
    assert(expected == C->get_data());
}

void test_mul() {
    t_tensor A = create_tensor(t_data({1, 2, 3}), t_shape({3}));
    t_tensor B = create_tensor(t_data({4, 5, 6}), t_shape({3}));
    t_tensor C = mul(A, B);
    t_data expected({4, 10, 18});
    assert(expected == C->get_data());
}

void test_matmul() {
    t_tensor A = create_tensor(t_data({1, 2, 3, 4}), t_shape({2, 2}));
    t_tensor B = create_tensor(t_data({5, 6, 7, 8}), t_shape({2, 2}));
    t_tensor C = matmul(A, B);
    t_data expected({19, 22, 43, 50});
    assert(expected == C->get_data());
}

void test_sum() {
    t_tensor A = create_tensor(t_data({1, 2, 3, 4}), t_shape({2, 2}));
    t_tensor S0 = sum(A, 0, false);
    t_tensor S1 = sum(A, 1, false);
    t_tensor Sall = sum(A, -1, false);
    t_data expected0({4, 6});
    t_data expected1({3, 7});
    t_data expected_all({10});
    // assert(expected0 == S0->get_data());
    // assert(expected1 == S1->get_data());
    // assert(expected_all == Sall->get_data());
}

void test_exp() {
    t_tensor A = create_tensor(t_data({0, 1}), t_shape({2}));
    t_tensor B = exp(A);
    t_data expected({std::exp(0.0f), std::exp(1.0f)});
    for (size_t i = 0; i < expected.size(); ++i)
        assert(std::abs(B->get_data()[i] - expected[i]) < 1e-5);
}

void test_log() {
    t_tensor A = create_tensor(t_data({1, std::exp(1.0f)}), t_shape({2}));
    t_tensor B = log(A);
    t_data expected({0.0f, 1.0f});
    for (size_t i = 0; i < expected.size(); ++i)
        assert(std::abs(B->get_data()[i] - expected[i]) < 1e-5);
}

void test_sigmoid() {
    t_tensor A = create_tensor(t_data({0}), t_shape({1}));
    t_tensor B = sigmoid(A);
    float expected = 1.0f / (1.0f + std::exp(-0.0f));
    assert(std::abs(B->get_data()[0] - expected) < 1e-5);
}

int main() {
    test_add();
    test_mul();
    test_matmul();
    test_sum();
    test_exp();
    test_log();
    test_sigmoid();
    return 0;
}