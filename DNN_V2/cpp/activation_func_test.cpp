#include <Eigen/Core>
#include "catch.hpp"
#include "activation_func.h"

using Eigen::MatrixXf;

TEST_CASE("sigmoid function is computed", "[sigmoid]") {
  MatrixXf Z(2,4);
  Z << -2.0f, -0.001f, 0.0002f, 2.0f, -1.23f, 0.0f, 0.74f, 2.6f;
  MatrixXf correct_answer(2,4);
  correct_answer << 0.119202922022118f, 0.499750000020833f, 0.500049999999833f, 0.880797077977882f,
                    0.226181425730546f, 0.500000000000000f, 0.676995856238523f, 0.930861579656653f;

  REQUIRE(correct_answer.isApprox(Sigmoid(Z), 1.0e-7));
}

TEST_CASE("relu function is computed", "[relu]") {
  MatrixXf Z(2,4);
  Z << -1.23f, -1e-7, 1e-8, 0.74f, -0.001f, 0.0f, 2e-4, 2.6f;
  MatrixXf correct_answer(2,4);
  correct_answer << 0.0f, 0.0f, 1e-8, 0.74f, 0.0f, 0.0f, 2e-4, 2.6f;

  REQUIRE(correct_answer.isApprox(Relu(Z), 1.0e-7));
}

TEST_CASE("derivative of sigmoid is computed", "[sigmoidBackward]") {
  MatrixXf A(2,4), dA(2,4);
  A << 0.2f, 1.3f, -0.3f, -1e-5, -0.001f, 2e-4, 1e-6, -1e-3;
  dA << 2.0f, -1.2f, -7.0f, 1.5f, -3.34f, 1.1f, 2.3f, -75.0f;
  MatrixXf correct_answer(2,4);
  correct_answer << 3.2e-1, 4.68e-1, 2.73f, -1.5e-5, 3.3433e-3, 2.1996e-4, 2.3e-6, 7.5075e-2;

  REQUIRE(correct_answer.isApprox(SigmoidBackward(dA, A), 1.0e-7));
}

TEST_CASE("derivative of relu is computed", "[reluBackward]") {
  MatrixXf Z(2,4), dA(2,4);
  Z << -1.23f, -1e-7, 1e-8, 0.74f, -0.001f, 0.0f, 2e-4, 2.6f;
  dA << 2.0f, -1.2f, -7.0f, 1.5f, -3.34f, 1.1f, 2.3f, -75.0f;
  MatrixXf correct_answer(2,4);
  correct_answer << 0.0f, 0.0f, -7.0f, 1.5f, 0.0f, 0.0f, 2.3f, -75.0f;

  REQUIRE(correct_answer.isApprox(ReluBackward(dA, Z), 1.0e-7));
}
