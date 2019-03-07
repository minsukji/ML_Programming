#include <Eigen/Core>
#include <vector>
#include <random>
#include "catch.hpp"
#include "dropout.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using std::vector;

TEST_CASE("check dropout", "[checkDropout]") {
  VectorXf layer_dropout(4);
  bool dropout;

  SECTION("case 1") {
    layer_dropout << 1.0f, 1.0f, 1.0f, 1.0f;
    dropout = CheckDropout(layer_dropout);
    REQUIRE(dropout == false);
  }

  SECTION("case 2") {
    layer_dropout << 0.7f, 1.0f, 1.0f, 1.0f;
    dropout = CheckDropout(layer_dropout);
    REQUIRE(dropout == true);
  }

  SECTION("case 3") {
    layer_dropout << 1.0f, 0.5f, 1.0f, 1.0f;
    dropout = CheckDropout(layer_dropout);
    REQUIRE(dropout == true);
  }

  SECTION("case 4") {
    layer_dropout << 1.0f, 1.0f, 1.0f, 0.4f;
    dropout = CheckDropout(layer_dropout);
    REQUIRE(dropout == true);
  }
}

TEST_CASE("compute dropout", "[dropout]") {
  // First hidden layer (3), Second hidden layer (2), Output layer (1)
  // Batch size (2)
  MatrixXf A1(3,2), A2(2,2), A3(1,2);
  A1 << 0.1f, -2.3f, 0.2f, 3.8f, 0.3f, 0.00012f;
  A2 << 11.2f, -0.004f, -20.7f, 1.6f;
  A3 << -16.4f, 1.1f;
  vector<MatrixXf> A;
  A.push_back(A1);
  A.push_back(A2);
  A.push_back(A3);

  SECTION("randomly select dropout") {
    unsigned int seed {1234UL};
    vector<MatrixXf> D = RandomlySelectDropout(A, seed);

    std::mt19937 mt(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float *correct_answer = new float[12] {};
    for (int i = 0; i < 12; ++i) {
      correct_answer[i] = uniform(mt);
    }

    REQUIRE(D[0](0,0)== Approx(correct_answer[0]));
    REQUIRE(D[0](1,0)== Approx(correct_answer[1]));
    REQUIRE(D[0](2,0)== Approx(correct_answer[2]));
    REQUIRE(D[0](0,1)== Approx(correct_answer[3]));
    REQUIRE(D[0](1,1)== Approx(correct_answer[4]));
    REQUIRE(D[0](2,1)== Approx(correct_answer[5]));
    REQUIRE(D[1](0,0)== Approx(correct_answer[6]));
    REQUIRE(D[1](1,0)== Approx(correct_answer[7]));
    REQUIRE(D[1](0,1)== Approx(correct_answer[8]));
    REQUIRE(D[1](1,1)== Approx(correct_answer[9]));
    REQUIRE(D[2](0,0)== Approx(correct_answer[10]));
    REQUIRE(D[2](0,1)== Approx(correct_answer[11]));
  }

  SECTION("apply dropout") {
    unsigned int seed {1234UL};
    float keep_prob = 0.7f;
    vector<MatrixXf> D = RandomlySelectDropout(A, seed); 
    ApplyDropout(D, A, keep_prob);
    MatrixXf R1(3,2), R2(2,2), R3(1,2);
    R1 << 0.142857142857143f, 0.0f, 0.285714285714286f, 5.428571428571429f, 0.428571428571429f, 0.000171428571429f;
    R2 << 0.0f, 0.0f, 0.0f, 0.0f;
    R3 << -23.42857142857143f, 1.57142857142857f;

    REQUIRE(A[0].isApprox(R1, 1.e-6));
    REQUIRE(A[1].isApprox(R2, 1.e-6));
    REQUIRE(A[2].isApprox(R3, 1.e-6));
  }

}
