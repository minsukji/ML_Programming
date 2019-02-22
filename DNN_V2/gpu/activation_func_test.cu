#include <limits>
#include "catch.hpp"
#include "activation_func.h"

extern const int nThreads {128};

TEST_CASE("sigmoid function is computed", "[sigmoid]") {
  int n {8};
  float *Z = new float[n] {-2.0f, -1.23f, -0.001f, 0.0f, 0.0002f, 0.74f, 2.0f, 2.6f};
  float *A = new float[n] {};
  float *d_Z, *d_A;

  cudaMalloc(&d_Z, n * sizeof(float));
  cudaMalloc(&d_A, n * sizeof(float));
  cudaMemcpy(d_Z, Z, n * sizeof(float), cudaMemcpyHostToDevice);

  int nBlocks {(n + nThreads - 1) / nThreads};
  Sigmoid<<<nBlocks, nThreads>>>(n, d_Z, d_A);

  cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_Z); cudaFree(d_A);

  float *correct_answer = new float[n] {0.119202922022118f, 0.226181425730546f, 0.499750000020833f, 0.500000000000000f,
                                        0.500049999999833f, 0.676995856238523f, 0.880797077977882f, 0.930861579656653f};
  for (int i = 0; i < n; ++i)
    REQUIRE(A[i] == Approx(correct_answer[i]).epsilon(std::numeric_limits<float>::epsilon()));

  delete[] Z; delete[] A; delete[] correct_answer;
}

TEST_CASE("relu function is computed", "[relu]") {
  int n {8};
  float *Z = new float[n] {-1.23f, -0.001f, -1e-7, 0.0f, 1e-8, 2e-4, 0.74f, 2.6f};
  float *A = new float[n] {};
  float *d_Z, *d_A;

  cudaMalloc(&d_Z, n * sizeof(float));
  cudaMalloc(&d_A, n * sizeof(float));
  cudaMemcpy(d_Z, Z, n * sizeof(float), cudaMemcpyHostToDevice);

  int nBlocks {(n + nThreads - 1) / nThreads};
  Relu<<<nBlocks, nThreads>>>(n, d_Z, d_A);

  cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_Z); cudaFree(d_A);

  float *correct_answer = new float[n] {0.0f, 0.0f, 0.0f, 0.0f, 1e-8, 2e-4, 0.74f, 2.6f};

  for (int i = 0; i < n; ++i)
    REQUIRE(A[i] == Approx(correct_answer[i]).epsilon(std::numeric_limits<float>::epsilon()));

  delete[] Z; delete[] A; delete[] correct_answer;
}
