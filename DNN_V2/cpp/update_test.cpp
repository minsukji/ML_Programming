#include <Eigen/Core>
#include <vector>
#include "catch.hpp"
#include "update.h"

using Eigen::MatrixXf;
using std::vector;

TEST_CASE("update parameters", "[updateParams]") {
  // Input layer (2); First hidden layer (4); Output layer (1)
  // W1(4,2), B1(4,1), W2(1,4), B2(1,1)
  // Batch_size (2)
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

  float learn_rate {0.1f};

  MatrixXf dW1(4,2), dW2(1,4), dB1(4,1), dB2(1,1);
  dW1 << -0.158519149308613f, -0.046908978362138f, 0.0f, 0.0f,
         0.195817772675345f, 0.057946385035582f, 1.678438051502957f, 0.496683300304992f;
  dW2 << -1.013773750122013f, 0.0f, -0.172459479272566f, -0.158854022308775f;
  dB1 << -0.193622381881916f, 0.0f, 0.239180589383544f, 2.050119337573233f;
  dB2 << -0.569477593770343f;

  vector<MatrixXf> grads;
  grads.push_back(dB2);
  grads.push_back(dW2);
  grads.push_back(dB1);
  grads.push_back(dW1);

  UpdateParameters(params, grads, learn_rate);

  MatrixXf new_W1(4,2), new_W2(1,4), new_B1(4,1), new_B2(1,1);
  new_W1 << 1.2158519149308613f, 2.8846908978362138f, -1.8f, 0.3f,
            -0.7195817772675345f, 0.8442053614964418f, -0.0278438051502957f, -0.6096683300304993f;
  new_W2 << 0.441377375012201f, 4.4f, -0.402754052072743f, -3.584114597769123f;
  new_B1 << 0.1193622381881916f, 0.2f, 0.6460819410616456f, 0.0949880662426767f;
  new_B2 << 0.206947759377034f;

  REQUIRE(params[0].isApprox(new_W1, 1.e-6));
  REQUIRE(params[1].isApprox(new_B1, 1.e-6));
  REQUIRE(params[2].isApprox(new_W2, 1.e-6));
  REQUIRE(params[3].isApprox(new_B2, 1.e-6));
}
