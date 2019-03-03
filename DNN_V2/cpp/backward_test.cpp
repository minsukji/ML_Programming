#include <Eigen/Core>
#include <vector>
#include "catch.hpp"
#include "backward.h"

TEST_CASE("backward propagation is computed", "[backwardProp]") {
  // Input layer (2); First hidden layer (4); Output layer (1)
  // W1(4,2), B1(4,1), W2(1,4), B2(1,1)
  // Batch_size (2)
  MatrixXf X(2,2);
  X << 0.7f, 0.9f, 0.45f, 0.1f;
  MatrixXi Y(1,2);
  Y << 1, 1;

  MatrixXf W1(4,2), W2(1,4), B1(4,1), B2(1,1);
  W1 << 1.2f, 2.88f, -1.8f, 0.3f, -0.7f, 0.85f, 0.14f, -0.56f;
  W2 << 0.34f, 4.4f, -0.42f, -3.6f;
  B1 << 0.1f, 0.2f, 0.67f, 0.3f;
  B2 << 0.15f;

  vector<MatrixXf> params;
  params.push_back(W1);
  params.push_back(B1);
  params.push_back(W2);
  params.push_back(B2);

  MatrixXf Z1(4,2), Z2(1,2), A1(4,2), A2(1,2);
  Z1 << 2.236f, 1.468f, -0.925f, -1.39f, 0.5625f, 0.125f, 0.146f, 0.37f;
  Z2 << 0.14839f, -0.73538f;
  A1 << 2.236f, 1.468f, 0.0f, 0.0f, 0.5625f, 0.125f, 0.146f, 0.37f;
  A2 << 0.537029576908463f, 0.324015235550852f;
  vector<MatrixXf> Z, A;
  Z.push_back(Z1);
  Z.push_back(Z2);
  A.push_back(A1);
  A.push_back(A2);

  SECTION("no regularization") {
    float lambda = 0.0f;

    vector<MatrixXf> grads = Backward(params, Z, A, X, Y, lambda);

    MatrixXf dW1(4,2), dW2(1,4), dB1(4,1), dB2(1,1);
    dW1 << -0.158519149308613f, -0.046908978362138f, 0.0f, 0.0f,
           0.195817772675345f, 0.057946385035582f, 1.678438051502957f, 0.496683300304992f;
    dW2 << -1.013773750122013f, 0.0f, -0.172459479272566f, -0.158854022308775;
    dB1 << -0.193622381881916f, 0.0f, 0.239180589383544f, 2.050119337573233f;
    dB2 << -0.569477593770343f;

    REQUIRE(grads[3].isApprox(dW1, 1.e-6));
    REQUIRE(grads[2].isApprox(dB1, 1.e-6));
    REQUIRE(grads[1].isApprox(dW2, 1.e-6));
    REQUIRE(grads[0].isApprox(dB2, 1.e-6));
  }

  SECTION("with L2 regularization") {
    float lambda = 0.1f;

    vector<MatrixXf> grads = Backward(params, Z, A, X, Y, lambda);

    MatrixXf dW1(4,2), dW2(1,4), dB1(4,1), dB2(1,1);

    dW1 << -0.0985191493086126f, 0.0970910216378619f, -0.09f, 0.015f,
           0.1608177726753449f, 0.1004463850355823f, 1.6854380515029568f, 0.4686833003049916f;
    dW2 << -0.996773750122013f, 0.22f, -0.193459479272566f, -0.338854022308775f;
    dB1 << -0.193622381881916f, 0.0f, 0.239180589383544f, 2.050119337573233f;
    dB2 << -0.569477593770343f;

    REQUIRE(grads[3].isApprox(dW1, 1.e-6));
    REQUIRE(grads[2].isApprox(dB1, 1.e-6));
    REQUIRE(grads[1].isApprox(dW2, 1.e-6));
    REQUIRE(grads[0].isApprox(dB2, 1.e-6));
  }
}
