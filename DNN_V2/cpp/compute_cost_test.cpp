#include <Eigen/Core>
#include <vector>
#include "catch.hpp"
#include "compute_cost.h"

TEST_CASE("compute costs", "[computeCosts]") {
  int n_layers {3};
  MatrixXf W1(2,2), W2(1,2), W3(1,1);
  W1 << 1.0325f, -0.124f, 0.0f, 0.0013f;
  W2 << 7.84f, -3.39f;
  W3 << -0.0001f;
  MatrixXf B1(1,2), B2(1,1), B3(1,1);
  B1 << 0.0f, 0.0f;
  B2 << 0.0f;
  B3 << 0.0f;
  vector<MatrixXf> params;
  params.push_back(W1);
  params.push_back(B1);
  params.push_back(W2);
  params.push_back(B2);
  params.push_back(W3);
  params.push_back(B3);

  int batch_size {5};
  MatrixXi Y(1,5);
  Y << 0, 1, 1, 1, 0;
  MatrixXf AL(1,5);
  AL << 0.1f, 0.97f, 0.51f, 0.3f, 0.02f;

  SECTION("without l2 regularization") {
    float lambda = 0.0f;
    float cost = ComputeCostBinaryClass(n_layers, batch_size, AL, Y, lambda, params);
    REQUIRE(cost == Approx(0.406667957609951f));
  }

  SECTION("with l2 regularization") {
    float lambda = 0.03f;
    float cost = ComputeCostBinaryClass(n_layers, batch_size, AL, Y, lambda, params);
    REQUIRE(cost == Approx(0.628785359459951));
  }
}
