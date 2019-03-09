#include <Eigen/Core>
#include <vector>
#include "catch.hpp"
#include "momentum.h"

using Eigen::MatrixXf;
using Eigen::VectorXi;
using std::vector;

TEST_CASE("momentum", "[momentum]") {
  // Input layer (3); First hidden layer (2); Output layer (1)
  // W1(2,3), B1(2,1), W2(1,2), B2(1,1)
  VectorXi layer_dims(3);
  layer_dims << 3, 2, 1;
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
  
  SECTION("initialize momentum") {
    vector<MatrixXf> V = InitializeMomentum(layer_dims);

    REQUIRE(V[0] == MatrixXf::Zero(1,1));
    REQUIRE(V[1] == MatrixXf::Zero(1,2));
    REQUIRE(V[2] == MatrixXf::Zero(2,1));
    REQUIRE(V[3] == MatrixXf::Zero(2,3));
  }

  SECTION("update momentum") {
    float learn_rate {0.12f};
    float beta {0.9f};

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

    UpdateMomentum(params, grads, V, learn_rate, beta);

    MatrixXf new_W1(2,3), new_B1(2,1), new_W2(1,2), new_B2(1,1);
    new_W1 << 1.196226038789295f, -0.699224689349739f, 0.286430621300522f, -1.80378f, 0.139244f, 0.8284f;
    new_B1 << 0.0839629125561356f, 0.19136f;
    new_W2 << -0.447082154888824f, 0.286f;
    new_B2 << 0.7294947990069007f;

    REQUIRE(params[0].isApprox(new_W1, 1.e-6));
    REQUIRE(params[1].isApprox(new_B1, 1.e-6));
    REQUIRE(params[2].isApprox(new_W2, 1.e-6));
    REQUIRE(params[3].isApprox(new_B2, 1.e-6));
  }
}
