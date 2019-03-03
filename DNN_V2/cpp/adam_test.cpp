#include <Eigen/Core>
#include <vector>
#include <tuple>
#include "catch.hpp"
#include "adam.h"

using Eigen::MatrixXf;
using std::vector;
using std::tuple;

TEST_CASE("adam", "[adam]") {
  // Input layer (3); First hidden layer (2); Output layer (1)
  // W1(2,3), B1(2,1), W2(1,2), B2(1,1)
  MatrixXf W1(2,3), B1(2,1), W2(1,2), B2(1,1);
  W1 << 1.2f, -0.7f, 0.3f, -1.8f, 0.14f, 0.85f;
  B1 << 0.1f, 0.2f;
  W2 << -0.56f, 0.34f;
  B2 << 0.67f;
  vector<MatrixXf> params;
  params.push_back(W1);
  params.push_back(B1);
  params.push_back(W2);
  params.push_back(B2);

  MatrixXf dW1(2,3), dB1(2,1), dW2(1,2), dB2(1,1);
  dW1 << 0.179496767558757f, 0.115390779144915f, 0.230781558289831f, 0.0f, 0.0f, 0.0f;
  dB1 << 0.256423953655368f, 0.0f;
  dW2 << -0.409820425931346f, 0.0f;
  dB2 << -0.457899917241728f;
  vector<MatrixXf> grads;
  grads.push_back(dB2);
  grads.push_back(dW2);
  grads.push_back(dB1);
  grads.push_back(dW1);

  SECTION("initialize adam") {
    tuple<vector<MatrixXf>, vector<MatrixXf>> result = InitializeAdam(grads);
    // V
    REQUIRE(std::get<0>(result)[0] == MatrixXf::Zero(1,1));
    REQUIRE(std::get<0>(result)[1] == MatrixXf::Zero(1,2));
    REQUIRE(std::get<0>(result)[2] == MatrixXf::Zero(2,1));
    REQUIRE(std::get<0>(result)[3] == MatrixXf::Zero(2,3));
    // S
    REQUIRE(std::get<1>(result)[0] == MatrixXf::Zero(1,1));
    REQUIRE(std::get<1>(result)[1] == MatrixXf::Zero(1,2));
    REQUIRE(std::get<1>(result)[2] == MatrixXf::Zero(2,1));
    REQUIRE(std::get<1>(result)[3] == MatrixXf::Zero(2,3));
  }

  SECTION("update adam") {
    float learn_rate {0.12f};
    float beta1 {0.9f};
    float beta2 {0.999f};
    int t {7};

    MatrixXf VdW1(2,3), VdB1(2,1), VdW2(1,2), VdB2(1,1);
    VdW1 << 0.015f, -0.02f, 0.1f, 0.035f, 0.007f, 0.2f;
    VdB1 << 0.12f, 0.08f;
    VdW2 << -1.0f, 0.5f;
    VdB2 << -0.5f;
    vector<MatrixXf> V;
    V.push_back(VdB2);
    V.push_back(VdW2);
    V.push_back(VdB1);
    V.push_back(VdW1);

    MatrixXf SdW1(2,3), SdB1(2,1), SdW2(1,2), SdB2(1,1);
    SdW1 << 0.0005f, 0.007f, 0.01f, 0.002f, 0.007f, 0.1f;
    SdB1 << 0.06f, 0.009f;
    SdW2 << 0.5f, 0.23f;
    SdB2 << 0.1f;
    vector<MatrixXf> S;
    S.push_back(SdB2);
    S.push_back(SdW2);
    S.push_back(SdB1);
    S.push_back(SdW1);

    UpdateAdam(params, grads, V, S, learn_rate, beta1, beta2, t);

    MatrixXf new_W1(2,3), new_B1(2,1), new_W2(1,2), new_B2(1,1);
    new_W1 << 1.173792200360343f, -0.698516781325965f, 0.278318123142131f,
              -1.813541548187539f, 0.138552347592999f, 0.839056776337345f;
    new_B1 << 0.0895165719322379f, 0.1854090352064458f;
    new_W2 << -0.534420261356016f, 0.321960629985865f;
    new_B2 << 0.7001103116296014f;

    REQUIRE(params[0].isApprox(new_W1, 1.e-6));
    REQUIRE(params[1].isApprox(new_B1, 1.e-6));
    REQUIRE(params[2].isApprox(new_W2, 1.e-6));
    REQUIRE(params[3].isApprox(new_B2, 1.e-6));
  }
}
