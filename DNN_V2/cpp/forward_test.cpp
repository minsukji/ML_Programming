#include <Eigen/Core>
#include <vector>
#include <tuple>
#include "catch.hpp"
#include "dropout.h"
#include "forward.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using std::vector;

TEST_CASE("forward propagation is computed", "[forwardProp]") {
  // Input layer (2); First hidden layer (4); Output layer (1)
  // W1(4,2), B1(4,1), W2(1,4), B2(1,1)
  // Batch size (2)
  VectorXi layer_dims(3);
  layer_dims << 2, 4, 1;
  int batch_size {2};

  MatrixXf X(2,2);
  X << 0.7f, 0.9f, 0.45f, 0.1f;

  MatrixXf W1(4,2), W2(1,4), B1(4,1), B2(1,1);
  W1 << 1.2f, 2.88f, -1.8f, 0.3f, -0.7f, 0.85f, 0.14f, -0.56f;
  W2 << 0.34f, 4.4f, -0.42f, -3.6f;
  B1 << 0.1f, 0.2f, 0.67f, -0.3f;
  B2 << 0.15f;
  vector<MatrixXf> params;
  params.push_back(W1);
  params.push_back(B1);
  params.push_back(W2);
  params.push_back(B2);

  SECTION("without dropout") {
    bool dropout {false}; 
    VectorXf layer_drop(3);
    layer_drop << 1.0f, 1.0f, 1.0f;
    vector<MatrixXf> D;

    tuple<vector<MatrixXf>, vector<MatrixXf>> result = Forward(X, params, dropout, layer_drop, D);
    vector<MatrixXf> Z = std::get<0>(result);
    vector<MatrixXf> A = std::get<1>(result);

    MatrixXf Z1(4,2), Z2(1,2), A1(4,2), A2(1,2);
    Z1 << 2.236f, 1.468f, -0.925f, -1.39f, 0.5625f, 0.125f, -0.454f, -0.23f;
    Z2 << 0.67399f, 0.59662f;
    A1 << 2.236f, 1.468f, 0.0f, 0.0f, 0.5625f, 0.125f, 0.0f, 0.0f;
    A2 << 0.662396010418219f, 0.644882635337264f;

    REQUIRE(Z[0].isApprox(Z1, 1.e-7));
    REQUIRE(Z[1].isApprox(Z2, 1.e-7));
    REQUIRE(A[0].isApprox(A1, 1.e-7));
    REQUIRE(A[1].isApprox(A2, 1.e-7));
  }

  SECTION("with dropout") {
    bool dropout {true};
    VectorXf layer_drop(3);
    layer_drop << 1.0f, 0.7f, 1.0f;
    unsigned int seed {1234UL};
    vector<MatrixXf> D = RandomlySelectDropout(layer_dims, batch_size, seed);

    tuple<vector<MatrixXf>, vector<MatrixXf>> result = Forward(X, params, dropout, layer_drop, D);
    vector<MatrixXf> Z = std::get<0>(result);
    vector<MatrixXf> A = std::get<1>(result);

    MatrixXf Z1(4,2), Z2(1,2), A1(4,2), A2(1,2);
    Z1 << 2.236f, 1.468f, -0.925f, -1.39, 0.5625f, 0.125f, -0.454f, -0.23f;
    A1 << 3.194285714285715f, 2.097142857142857f, 0.0f, 0.0f, 0.803571428571429f, 0.0f, 0.0f, 0.0f;
    Z2 << 0.898557142857143f, 0.863028571428572f;
    A2 << 0.710652904814677f, 0.703293021100006f;

    REQUIRE(Z[0].isApprox(Z1, 1.e-7));
    REQUIRE(Z[1].isApprox(Z2, 1.e-7));
    REQUIRE(A[0].isApprox(A1, 1.e-7));
    REQUIRE(A[1].isApprox(A2, 1.e-7));
  }
}
