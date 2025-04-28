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
    t_tensor A = create_tensor(t_data({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    }), t_shape({2, 3, 4}));
    
    // Create input B of shape [4, 5]
    t_tensor B = create_tensor(t_data({
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    }), t_shape({4, 5}));
    
    // Run matmul: A [2,3,4] @ B [4,5] -> [2,3,5]
    t_tensor C = matmul(A, B);
    
    // Expected result: manually computed or verified via NumPy
    t_data expected({
        110, 120, 130, 140, 150,
        246, 272, 298, 324, 350,
        382, 424, 466, 508, 550,
    
        518, 576, 634, 692, 750,
        654, 728, 802, 876, 950,
        790, 880, 970, 1060, 1150
    });
    assert(C->get_shape() == t_shape({2, 3, 5}));
    assert(C->get_data() == expected);
}

void test_sum() {
    t_tensor A = create_tensor(t_data({1, 2, 3, 4}), t_shape({2, 2}));
    t_tensor S0 = sum(A, 0, false);
    t_tensor S1 = sum(A, 1, false);
    t_tensor Sall = sumall(A);
    t_data expected0({4, 6});
    t_data expected1({3, 7});
    t_data expected_all({10});
    assert(expected0 == S0->get_data());
    assert(expected1 == S1->get_data());
    assert(expected_all == Sall->get_data());
}

void test_exp() {
    t_tensor A = create_tensor(t_data({0, 1}), t_shape({2}));
    t_tensor B = exp(A);
    t_data expected({std::exp(0.0f), std::exp(1.0f)});
    for (size_t i = 0; i < expected.size(); i++)
        assert(std::abs(B->get_data()[i] - expected[i]) < 1e-5);
}

void test_log() {
    t_tensor A = create_tensor(t_data({1, std::exp(1.0f)}), t_shape({2}));
    t_tensor B = log(A);
    t_data expected({0.0f, 1.0f});
    for (size_t i = 0; i < expected.size(); i++)
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