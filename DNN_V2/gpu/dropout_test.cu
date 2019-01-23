#include <limits>
#include "catch2/catch.hpp"
#include "dropout.h"

extern const int nThreads {128};

TEST_CASE("randomly initialize dropout", "[randomInitDropout]") {
  int n {12};
  float *D = new float[n] {};
  float *d_D;
  cudaMalloc(&d_D, n * sizeof(float));
  RandomlySelectDropout(n, d_D);
  cudaMemcpy(D, d_D, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_D);

  float *correct_answer = new float[n] {0.147921f, 0.460329f, 0.30005f,
                                        0.907808f, 0.0552578f, 0.0109106f,
                                        0.251024f, 0.263781f, 0.391988f,
                                        0.72534f, 0.0967395f, 0.75908f};
  for (int i = 0; i < n; ++i)
    REQUIRE(D[i] == Approx(correct_answer[i]));

  delete[] D; delete[] correct_answer;
}

TEST_CASE("apply dropout", "[applyDropout]") {
  int n {12};
  float *D = new float[n] {0.147921f, 0.460329f, 0.30005f,
                          0.907808f, 0.0552578f, 0.0109106f,
                          0.251024f, 0.263781f, 0.391988f,
                          0.72534f, 0.0967395f, 0.75908f};

  float *A = new float[n] {0.1f, 0.2f, 0.3f, -2.3f, 3.8f, 0.00012f,
                           11.2f, -20.7f, -0.004f, 1.6f, -16.4f, 1.1f};
  float *d_D, *d_A;
  cudaMalloc(&d_D, n * sizeof(float));
  cudaMalloc(&d_A, n * sizeof(float));
  cudaMemcpy(d_D, D, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, A, n * sizeof(float), cudaMemcpyHostToDevice);

  float keep_prob = 0.7f;
  int nBlocks = (n + nThreads - 1) / nThreads;
  ApplyDropout<<<nBlocks, nThreads>>>(n, d_D, d_A, keep_prob);
  cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_D); cudaFree(d_A);

  float *correct_answer = new float[n] {0.142857142857143f, 0.285714285714286f,
                                        0.428571428571429f, 0.0f,
                                        5.42857142857143f, 1.71428571428571e-04,
                                        16.0f, -29.5714285714286f,
                                        -0.00571428571428572f, 0.0f,
                                        -23.4285714285714f, 0.0f};

  for (int i = 0; i < n; ++i)
    REQUIRE(A[i] == Approx(correct_answer[i]));

  delete[] D; delete[] A; delete[] correct_answer;
}
