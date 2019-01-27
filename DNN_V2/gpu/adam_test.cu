#include "catch2/catch.hpp"
#include "adam.h"
#include <iostream>

extern const int nThreads {128};

TEST_CASE("initialize adam parameters", "[initAdam]") {
  int n_layers {2};
  int *W_index = new int[n_layers+1] {0, 6, 8};
  int *B_index = new int[n_layers+1] {0, 2, 3};
  int n_W {W_index[n_layers]};
  int n_B {B_index[n_layers]};
  float *VdW = new float[n_W] {};
  float *SdW = new float[n_W] {};
  float *VdB = new float[n_B] {};
  float *SdB = new float[n_B] {};
  float *d_VdW, *d_SdW, *d_VdB, *d_SdB;
  cudaMalloc(&d_VdW, n_W * sizeof(float));
  cudaMalloc(&d_SdW, n_W * sizeof(float));
  cudaMalloc(&d_VdB, n_B * sizeof(float));
  cudaMalloc(&d_SdB, n_W * sizeof(float));
  int nBlocks {0};

  SECTION("initialize VdB and SdB") {
    nBlocks = (n_B + nThreads - 1) / nThreads;
    InitializeVdBSdB<<<nBlocks, nThreads>>>(n_B, d_VdB, d_SdB);
    cudaMemcpy(VdB, d_VdB, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(SdB, d_SdB, n_B * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_VdB = new float[n_B] {0.0f, 0.0f, 0.0f};
    float *correct_SdB = new float[n_B] {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_B; ++i) {
      REQUIRE(VdB[i] == Approx(correct_VdB[i]));
      REQUIRE(SdB[i] == Approx(correct_SdB[i]));
    }
    delete[] correct_VdB; delete[] correct_SdB;
  }

  SECTION("initialize VdW and SdW") {
    nBlocks = (n_W + nThreads - 1) / nThreads;
    InitializeVdWSdW<<<nBlocks, nThreads>>>(n_W, d_VdW, d_SdW);
    cudaMemcpy(VdW, d_VdW, n_W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(SdW, d_SdW, n_W * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_VdW = new float[n_W] {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    float *correct_SdW = new float[n_W] {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_W; ++i) {
      REQUIRE(VdW[i] == Approx(correct_VdW[i]));
      REQUIRE(SdW[i] == Approx(correct_SdW[i]));
    }
    delete[] correct_VdW; delete[] correct_SdW;
  }

  SECTION("initialize VdB, SdB, VdW, and SdW") {
    InitializeAdam(n_layers, d_VdW, d_SdW, d_VdB, d_SdB,
                   W_index, B_index);
    cudaMemcpy(VdB, d_VdB, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(SdB, d_SdB, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VdW, d_VdW, n_W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(SdW, d_SdW, n_W * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_VdB = new float[n_B] {0.0f, 0.0f, 0.0f};
    float *correct_SdB = new float[n_B] {0.0f, 0.0f, 0.0f};
    float *correct_VdW = new float[n_W] {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    float *correct_SdW = new float[n_W] {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_B; ++i) {
      REQUIRE(VdB[i] == Approx(correct_VdB[i]));
      REQUIRE(SdB[i] == Approx(correct_SdB[i]));
    }
    for (int i = 0; i < n_W; ++i) {
      REQUIRE(VdW[i] == Approx(correct_VdW[i]));
      REQUIRE(SdW[i] == Approx(correct_SdW[i]));
    }
    delete[] correct_VdB; delete[] correct_SdB;
    delete[] correct_VdW; delete[] correct_SdW;
  }

  cudaFree(d_VdW); cudaFree(d_SdW); cudaFree(d_VdB); cudaFree(d_SdB);
  delete[] VdW; delete[] SdW; delete[] VdB; delete[] SdB;
  delete[] W_index; delete[] B_index;
}

TEST_CASE("update adam parameters", "[updateAdam]") {
  int n_W {8};
  float *VdW = new float[n_W] {0.015f, 0.035f, -0.02f, 0.007f,
                               0.1f, 0.2f, -1.0f, 0.5f};
  float *SdW = new float[n_W] {0.0005f, 0.002f, 0.007f, 0.007f,
                               0.01f, 0.1f, 0.5f, 0.23f};
  float *W = new float[n_W] {1.2f, -1.8f, -0.7f, 0.14f, 0.3f, 0.85f, -0.56f, 0.34f};
  float *dW = new float[n_W] {0.179496767558757f, 0.0f, 0.115390779144915f, 0.0f,
                              0.230781558289831f, 0.0f, -0.409820425931346f, 0.0f};
 
  int n_B {3};
  float *VdB = new float[n_B] {0.12f, 0.08f, -0.5f};
  float *SdB = new float[n_B] {0.06f, 0.009f, 0.1f};
  float *B = new float[n_B] {0.1f, 0.2f, 0.67f};
  float *dB = new float[n_B] {0.256423953655368f, 0.0f, -0.457899917241728f};

  float *d_VdW, *d_SdW, *d_W, *d_dW, *d_VdB, *d_SdB, *d_B, *d_dB;
  cudaMalloc(&d_VdW, n_W * sizeof(float));
  cudaMalloc(&d_SdW, n_W * sizeof(float));
  cudaMalloc(&d_W, n_W * sizeof(float));
  cudaMalloc(&d_dW, n_W * sizeof(float));
  cudaMalloc(&d_VdB, n_B * sizeof(float));
  cudaMalloc(&d_SdB, n_B * sizeof(float));
  cudaMalloc(&d_B, n_B * sizeof(float));
  cudaMalloc(&d_dB, n_B * sizeof(float));

  cudaMemcpy(d_VdW, VdW, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_SdW, SdW, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dW, dW, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_VdB, VdB, n_B * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_SdB, SdB, n_B * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n_B * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dB, dB, n_B * sizeof(float), cudaMemcpyHostToDevice);

  float beta1 {0.9f};
  float beta2 {0.999f};
  float learn_rate {0.12f};
  int nBlocks {0};
  int t {7};

  SECTION("update adam B") {
    nBlocks = (n_B + nThreads - 1) / nThreads;
    UpdateAdamB<<<nBlocks, nThreads>>>(n_B, d_VdB, d_SdB, d_B, d_dB,
                                       beta1, beta2, learn_rate, t);
    cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_B = new float[n_B] {0.0895165719322379f,
                                       0.1854090352064458f,
                                       0.7001103116296014f};
    for (int i = 0; i < n_B; ++i)
      REQUIRE(B[i] == Approx(correct_B[i]));

    delete[] correct_B;
  }

  SECTION("update adam W") {
    nBlocks = (n_W + nThreads - 1) / nThreads;
    UpdateAdamW<<<nBlocks, nThreads>>>(n_W, d_VdW, d_SdW, d_W, d_dW,
                                       beta1, beta2, learn_rate, t);
    cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_W = new float[n_W] {1.173792200360343f, -1.813541548187539f,
                                       -0.698516781325965f, 0.138552347592999f,
                                       0.278318123142131f, 0.839056776337345f,
                                       -0.534420261356016, 0.321960629985865f};
    for (int i = 0; i < n_W; ++i)
      REQUIRE(W[i] == Approx(correct_W[i]));

    delete[] correct_W;
  }

  SECTION("update adam W and B") {
    UpdateAdam(n_W, d_VdW, d_SdW, d_W, d_dW, n_B, d_VdB, d_SdB, d_B, d_dB,
               beta1, beta2, learn_rate, t);
    cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_B = new float[n_B] {0.0895165719322379f,
                                       0.1854090352064458f,
                                       0.7001103116296014f};
    float *correct_W = new float[n_W] {1.173792200360343f, -1.813541548187539f,
                                       -0.698516781325965f, 0.138552347592999f,
                                       0.278318123142131f, 0.839056776337345f,
                                       -0.534420261356016, 0.321960629985865f};
    for (int i = 0; i < n_B; ++i)
      REQUIRE(B[i] == Approx(correct_B[i]));
    for (int i = 0; i < n_W; ++i)
      REQUIRE(W[i] == Approx(correct_W[i]));

    delete[] correct_B; delete[] correct_W; 
  }

  cudaFree(d_VdW); cudaFree(d_SdW); cudaFree(d_W); cudaFree(d_dW);
  cudaFree(d_VdB); cudaFree(d_SdB); cudaFree(d_B); cudaFree(d_dB);

  delete[] VdW; delete[] SdW; delete[] W; delete[] dW;
  delete[] VdB; delete[] SdB; delete[] B; delete[] dB;
}
