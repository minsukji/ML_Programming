#include <limits>
#include "catch2/catch.hpp"
#include "update.h"

extern const int nThreads {128};

TEST_CASE("update parameters", "[updateParams]") {
  int n_W {8};
  float *W = new float[n_W] {1.2f, -1.8f, -0.7f, 0.14f, 0.3f, 0.85f, -0.56f, 0.34f};
  float *dW = new float[n_W] {0.179496767558757f, 0.0f, 0.115390779144915f, 0.0f,
                              0.230781558289831f, 0.0f, -0.409820425931346f, 0.0f};
  int n_B {3};
  float *B = new float[n_B] {0.1f, 0.2f, 0.67f};
  float *dB = new float[n_B] {0.256423953655368f, 0.0f, -0.457899917241728f};

  float *d_W, *d_dW, *d_B, *d_dB;
  cudaMalloc(&d_W, n_W * sizeof(float));
  cudaMalloc(&d_dW, n_W * sizeof(float));
  cudaMalloc(&d_B, n_B * sizeof(float));
  cudaMalloc(&d_dB, n_B * sizeof(float));

  cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dW, dW, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n_B * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dB, dB, n_B * sizeof(float), cudaMemcpyHostToDevice);

  float learn_rate {0.1f};

  float *correct_B = new float[n_B] {0.0743576046344633f, 0.2f, 0.715789991724173f};
  float *correct_W = new float[n_W] {1.182050323244124f, -1.8f, -0.711539077914491f, 0.14f,
                                     0.276921844171017f, 0.85f, -0.519017957406865f, 0.34f};

  int nBlocks {0};

  SECTION("update B") {
    nBlocks = (n_B + nThreads - 1) / nThreads;
    UpdateB<<<nBlocks, nThreads>>>(n_B, d_B, d_dB, learn_rate);
    cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_B; ++i)
      CHECK(B[i] == Approx(correct_B[i]).epsilon(std::numeric_limits<float>::epsilon()));
  }

  SECTION("update W") {
    nBlocks = (n_W + nThreads - 1) / nThreads;
    UpdateW<<<nBlocks, nThreads>>>(n_W, d_W, d_dW, learn_rate);
    cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_W; ++i)
      CHECK(W[i] == Approx(correct_W[i]).epsilon(std::numeric_limits<float>::epsilon()));
  }

  SECTION("update W and B") {
    UpdateParameters(n_W, d_W, d_dW, n_B, d_B, d_dB, learn_rate);
    cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_B; ++i)
      CHECK(B[i] == Approx(correct_B[i]).epsilon(std::numeric_limits<float>::epsilon()));

    cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_W; ++i)
      CHECK(W[i] == Approx(correct_W[i]).epsilon(std::numeric_limits<float>::epsilon()));
  }

  cudaFree(d_W); cudaFree(d_dW); cudaFree(d_B); cudaFree(d_dB);
  delete[] W; delete[] dW; delete[] B; delete[] dB;
  delete[] correct_W; delete[] correct_B;
}
