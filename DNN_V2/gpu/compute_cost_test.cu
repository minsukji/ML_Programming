#include <limits>
#include "catch2/catch.hpp"
#include "compute_cost.h"

extern const int nThreads {128};

TEST_CASE("cross entropy element-wise compute", "[crossEntropyElements]") {
  int n {9};

  // Elements 0 to 4 are ideal cases
  // Elements 5 and 6 are non-ideal cases where cost is negative
  // Elements 7 and 8 are cases with NaN: i.e. logf(negative number)
  float *AL = new float[n] {0.1f, 0.97f, 0.51f, 0.3f, 0.02f,
                            -1e-4, 1.0001f, 1.0001f, -1e-4};
  int *Y = new int[n] {0, 1, 1, 1, 0, 0, 1, 0, 1};
  float *cost_temp = new float[n] {};

  float *d_AL, *d_cost_temp;
  int *d_Y;
  cudaMalloc(&d_AL, n * sizeof(float));
  cudaMalloc(&d_cost_temp, n * sizeof(float));
  cudaMalloc(&d_Y, n * sizeof(int));
  cudaMemcpy(d_AL, AL, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, Y, n * sizeof(int), cudaMemcpyHostToDevice);

  int nBlocks = (n + nThreads - 1) / nThreads;
  CrossEntropyElementWiseCompute<<<nBlocks, nThreads>>>(n, d_AL, d_Y, d_cost_temp);
  cudaMemcpy(cost_temp, d_cost_temp, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_AL); cudaFree(d_cost_temp); cudaFree(d_Y);

  float *correct_answer = new float[n] {0.105360515657826f, 0.0304592074847086f,
                                        0.673344553263766f, 1.20397280432594f,
                                        0.0202027073175195f, -9.99950003332973e-05,
                                        -9.99950003332973e-05,
                                        std::numeric_limits<float>::quiet_NaN(),
                                        std::numeric_limits<float>::quiet_NaN()};

  for (int i = 0; i < 7; ++i)
    REQUIRE(cost_temp[i]
            == Approx(correct_answer[i]).margin(std::numeric_limits<float>::epsilon()));

  // Check for NaN: i.e. logf(negative number) 
  REQUIRE(std::isnan(cost_temp[7]) == true);
  REQUIRE(std::isnan(cost_temp[8]) == true); 

  delete[] AL; delete[] Y; delete[] cost_temp; delete[] correct_answer;
}

TEST_CASE("cross entropy cost", "[crossEntropyCost]") {
  int n {5};
  float *AL = new float[n] {0.1f, 0.97f, 0.51f, 0.3f, 0.02f};
  int *Y = new int[n] {0, 1, 1, 1, 0};
  float *cost_temp = new float[n] {};
  float *cost = new float {0.0f};
  float *d_AL, *d_cost_temp;
  int *d_Y;
  cudaMalloc(&d_AL, n * sizeof(float));
  cudaMalloc(&d_cost_temp, n * sizeof(float));
  cudaMalloc(&d_Y, n * sizeof(int));
  cudaMemcpy(d_AL, AL, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, Y, n * sizeof(int), cudaMemcpyHostToDevice);
  CostCrossEntropy(n, d_AL, d_Y, d_cost_temp, cost);

  float correct_answer = 0.40666795761f;
  REQUIRE(*cost == Approx(correct_answer).margin(std::numeric_limits<float>::epsilon()));

  cudaFree(d_AL); cudaFree(d_Y); cudaFree(d_cost_temp);
  delete[] AL; delete[] Y; delete[] cost_temp; delete cost;
}

TEST_CASE("l2 regularization cost", "[l2RegCost]") {
  int batch_size {8};
  int n_W {7};
  float *W = new float[n_W] {1.0325f, -0.124f, 0.0f, 0.0013f, 7.84f, -3.39f, -0.0001f};
  float lambda = 0.03;
  float *cost = new float {0.0f};
  float *d_W;
  cudaMalloc(&d_W, n_W * sizeof(float));
  cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);
  CostL2Regularization(batch_size, n_W, d_W, lambda, cost);
  cudaFree(d_W);

  float correct_answer = 0.138823376156250f;
  REQUIRE(*cost == Approx(correct_answer).margin(std::numeric_limits<float>::epsilon()));
  delete[] W; delete cost;
}

TEST_CASE("compare costs from gpu and serial computations", "[compareCosts]") {
  int batch_size {5};
  float *AL = new float[batch_size] {0.1f, 0.97f, 0.51f, 0.3f, 0.02f};
  int *Y = new int[batch_size] {0, 1, 1, 1, 0};
  float *cost_temp = new float[batch_size] {};
  float *d_AL, *d_cost_temp;
  int *d_Y;
  cudaMalloc(&d_AL, batch_size * sizeof(float));
  cudaMalloc(&d_Y, batch_size * sizeof(int));
  cudaMalloc(&d_cost_temp, batch_size * sizeof(float));
  cudaMemcpy(d_AL, AL, batch_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, Y, batch_size * sizeof(int), cudaMemcpyHostToDevice);

  int n_W {7};
  float *W = new float[n_W] {1.0325f, -0.124f, 0.0f, 0.0013f, 7.84f, -3.39f, -0.0001f};
  float *d_W;
  cudaMalloc(&d_W, n_W * sizeof(float));
  cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);
  float *gpu_cost = new float {0.0f};
  float *cpu_cost = new float {0.0f};

  SECTION("without l2 regularization") {
    float lambda {0.0f};
    ComputeCostBinaryClass(batch_size, d_AL, d_Y, d_cost_temp, lambda, n_W, d_W, gpu_cost);
    ComputeCostBinaryClassSerial(batch_size, d_AL, d_Y, lambda, n_W, d_W, cpu_cost);
    REQUIRE(*gpu_cost == *cpu_cost);
  }

  SECTION("with l2 regularization") {
    float lambda {0.03f};
    ComputeCostBinaryClass(batch_size, d_AL, d_Y, d_cost_temp, lambda, n_W, d_W, gpu_cost);
    ComputeCostBinaryClassSerial(batch_size, d_AL, d_Y, lambda, n_W, d_W, cpu_cost);
    REQUIRE(*gpu_cost == *cpu_cost);
  }

  cudaFree(d_AL); cudaFree(d_Y); cudaFree(d_cost_temp); cudaFree(d_W);
  delete[] AL; delete[] Y; delete[] cost_temp; delete[] W; delete[] gpu_cost; delete[] cpu_cost;
}

TEST_CASE("compute accuracy", "[accuracy]") {
  int n {10};
  float *predict_prob = new float[n] {0.1f, 0.9f, 0.501f, 0.499f, 0.499f, 0.501f, 0.4f, 0.65f, 0.5f, 0.1f};
  int *out_bin = new int[n] {0, 1, 1, 0, 1, 0, 0, 1, 1, 1};
  float thres {0.5f};

  float accuracy = ComputeAccuracyBinaryClass(n, predict_prob, out_bin, thres);
  float correct_accuracy = 0.7f;
  REQUIRE(accuracy == Approx(correct_accuracy));

  delete[] predict_prob;
  delete[] out_bin;
}
