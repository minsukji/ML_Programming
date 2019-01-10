#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "dnn_engine.h"
#include <gtest/gtest.h>
#include <array>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using std::vector;
using std::tuple;
using std::array;

TEST(SigmoidTest, Case_1)
{
    MatrixXd z(2,3);
    z <<  0.0, -1.23, 0.74, -0.001, 2.6, 0.0002;
    MatrixXd expectedResult(2,3);
    expectedResult <<  0.5, 0.22618142573, 0.67699585624, 0.49975, 0.930861579657, 0.50005;
    ASSERT_TRUE(expectedResult.isApprox(Sigmoid(z), 1.0e-8));
}

TEST(ReluTest, Case_1)
{
    MatrixXd z(2,3);
    z <<  0.0, -1.23, 0.74, -0.001, 2.6, 0.0002;
    MatrixXd expectedResult(2,3);
    expectedResult << 0.0, 0.0, 0.74, 0.0, 2.6, 0.0002;
    ASSERT_TRUE(expectedResult.isApprox(Relu(z), 1.0e-8));
}

TEST(InitParamsTest, DimensionCheck_1)
{
    // Input layer (3); First hidden layer (4); Second hidden layer (8); Output layer (1)
    VectorXi layer_dims(4);
    layer_dims << 3, 4, 8, 1;
    vector<MatrixXd> params = init_params(layer_dims);

    ASSERT_EQ(6, params.size());
    ASSERT_EQ(std::make_tuple(4, 3), std::make_tuple(params[0].rows(), params[0].cols()));
    ASSERT_EQ(std::make_tuple(4, 1), std::make_tuple(params[1].rows(), params[1].cols()));
    ASSERT_EQ(std::make_tuple(8, 4), std::make_tuple(params[2].rows(), params[2].cols()));
    ASSERT_EQ(std::make_tuple(8, 1), std::make_tuple(params[3].rows(), params[3].cols()));
    ASSERT_EQ(std::make_tuple(1, 8), std::make_tuple(params[4].rows(), params[4].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[5].rows(), params[5].cols()));
}

TEST(InitParamsTest, DimensionCheck_2)
{
    // Input layer (1); First hidden layer (1); Second hidden layer (1); Output layer (1)
    VectorXi layer_dims(4);
    layer_dims << 1, 1, 1, 1;
    vector<MatrixXd> params = init_params(layer_dims);

    ASSERT_EQ(6, params.size());
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[0].rows(), params[0].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[1].rows(), params[1].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[2].rows(), params[2].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[3].rows(), params[3].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[4].rows(), params[4].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[5].rows(), params[5].cols()));
}

TEST(InitParamsTest, DimensionCheck_3)
{
    // Input layer (1); First hidden layer (1); Output layer (10)
    VectorXi layer_dims(3);
    layer_dims << 1, 1, 10;
    vector<MatrixXd> params = init_params(layer_dims);

    ASSERT_EQ(4, params.size());
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[0].rows(), params[0].cols()));
    ASSERT_EQ(std::make_tuple(1, 1), std::make_tuple(params[1].rows(), params[1].cols()));
    ASSERT_EQ(std::make_tuple(10, 1), std::make_tuple(params[2].rows(), params[2].cols()));
    ASSERT_EQ(std::make_tuple(10, 1), std::make_tuple(params[3].rows(), params[3].cols()));
}

/*TEST(InitParamsTest, DimensionCheck_4)
{
    VectorXi layer_dims(2);
    layer_dims << 2, 3;
    ASSERT_THROW(init_params(layer_dims), char*);
}
*/

TEST(ForwardTest, Case_1)
{
    // Input layer (3); First hidden layer (2); Output layer (1)
    // W1(2, 3), B1(2, 1), W2(1, 2), B2(1, 1)
    MatrixXd X(3, 1);
    X << 0.7, 0.45, 0.9;
    vector<MatrixXd> params;
    MatrixXd W(2, 3);
    W << 1.2, -0.7, 0.3, -1.8, 0.14, 0.85;
    params.push_back(W);
    MatrixXd B(2, 1);
    B << 0.1, 0.2;
    params.push_back(B);
    W.resize(1, 2);
    W << -0.56, 0.34;
    params.push_back(W);
    B.resize(1, 1);
    B << 0.67;
    params.push_back(B);
   
    tuple<vector<MatrixXd>, vector<MatrixXd>> result = forward(X, params);
    vector<MatrixXd> Z = std::get<0>(result);
    vector<MatrixXd> A = std::get<1>(result);
    
    MatrixXd Z1(2, 1);
    Z1 << 0.895, -0.232;
    MatrixXd A1(2, 1);
    A1 << 0.895, 0.0;
    MatrixXd Z2(1, 1);
    Z2 << 0.16880;
    MatrixXd A2(1, 1);
    A2 << 0.542100082758272;

    ASSERT_TRUE(Z1.isApprox(Z[0], 1.e-8));
    ASSERT_TRUE(Z2.isApprox(Z[1], 1.e-8));
    ASSERT_TRUE(A1.isApprox(A[0], 1.e-8));
    ASSERT_TRUE(A2.isApprox(A[1], 1.e-8));
}

TEST(ComputeCostTest, case_1)
{
    MatrixXd AL(1, 5);
    AL << 0.1, 0.97, 0.51, 0.3, 0.02;
    MatrixXi Y(1, 5);
    Y << 0, 1, 1, 1, 0;
    double expectedCost = 0.40666795761;
    ASSERT_NEAR(expectedCost, compute_cost(AL, Y), 1.0e-8);
}

TEST(BackwardTest, case_1)
{
    MatrixXd X(3, 1);
    X << 0.7, 0.45, 0.9;
    vector<MatrixXd> params;
    MatrixXd W(2, 3);
    W << 1.2, -0.7, 0.3, -1.8, 0.14, 0.85;
    params.push_back(W);
    MatrixXd B(2, 1);
    B << 0.1, 0.2;
    params.push_back(B);
    W.resize(1, 2);
    W << -0.56, 0.34;
    params.push_back(W);
    B.resize(1, 1);
    B << 0.67;
    params.push_back(B);
    MatrixXi Y(1,1);
    Y << 1;
    tuple<vector<MatrixXd>, vector<MatrixXd>> result = forward(X, params);
    vector<MatrixXd> Z = std::get<0>(result);
    vector<MatrixXd> A = std::get<1>(result);

    vector<MatrixXd> grads = backward(params, Z, A, X, Y);

    MatrixXd dB2(1,1), dW2(1,2), dB1(2,1), dW1(2,3);
    dB2 << -0.457899917241728;
    dW2 << -0.409820425931346, 0.0;
    dB1 << 0.256423953655368, 0.0;
    dW1 << 0.179496767558757, 0.115390779144915, 0.230781558289831, 0.0, 0.0, 0.0;

    ASSERT_TRUE(dB2.isApprox(grads[0], 1.e-10));
    ASSERT_TRUE(dW2.isApprox(grads[1], 1.e-10));
    ASSERT_TRUE(dB1.isApprox(grads[2], 1.e-10));
    ASSERT_TRUE(dW1.isApprox(grads[3], 1.e-10));
}

TEST(UpdateParamsTest, Case_1)
{
    vector<MatrixXd> params, grads;
    MatrixXd W1(2,3), B1(2,1), W2(1,2), B2(1,1);
    MatrixXd dW1(2,3), dB1(2,1), dW2(1,2), dB2(1,1);
    MatrixXd new_W1(2,3), new_B1(2,1), new_W2(1,2), new_B2(1,1);
    W1 << 1.2, -0.7, 0.3, -1.8, 0.14, 0.85;
    params.push_back(W1);
    B1 << 0.1, 0.2;
    params.push_back(B1);
    W2 << -0.56, 0.34;
    params.push_back(W2);
    B2 << 0.67;
    params.push_back(B2);

    dB2 << -0.457899917241728;
    grads.push_back(dB2);
    dW2 << -0.409820425931346, 0.0;
    grads.push_back(dW2);
    dB1 << 0.256423953655368, 0.0;
    grads.push_back(dB1);
    dW1 << 0.179496767558757, 0.115390779144915, 0.230781558289831, 0.0, 0.0, 0.0;
    grads.push_back(dW1);

    vector<MatrixXd> new_params = update_params(params, grads, 0.1);

    new_W1 << 1.182050323244124, -0.711539077914491, 0.276921844171017,
             -1.8, 0.14, 0.85;
    new_B1 << 0.0743576046344633, 0.2;
    new_W2 << -0.519017957406865, 0.34;
    new_B2 << 0.715789991724173;

    ASSERT_TRUE(new_W1.isApprox(new_params[0], 1.e-10));
    ASSERT_TRUE(new_B1.isApprox(new_params[1], 1.e-10));
    ASSERT_TRUE(new_W2.isApprox(new_params[2], 1.e-10));
    ASSERT_TRUE(new_B2.isApprox(new_params[3], 1.e-10));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    //return 0;
}
