#include "catch.hpp"
#include "momentum.h"

extern const int nThreads {128};

TEST_CASE("initialize momentum parameters", "[initMomentum]") {
  int n_layers {2};
  int *W_index = new int[n_layers+1] {0, 6, 8};
  int *B_index = new int[n_layers+1] {0, 2, 3};
  int n_W {W_index[n_layers]};
  int n_B {B_index[n_layers]};
  float *VdW = new float[n_W] {};
  float *VdB = new float[n_B] {};
  float *d_VdW, *d_VdB;
  cudaMalloc(&d_VdW, n_W * sizeof(float));
  cudaMalloc(&d_VdB, n_B * sizeof(float));
  int nBlocks {0};

  SECTION("initialize VdB") {
    nBlocks = (n_B + nThreads - 1) / nThreads;
    InitializeVdB<<<nBlocks, nThreads>>>(n_B, d_VdB);
    cudaMemcpy(VdB, d_VdB, n_B * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_VdB = new float[n_B] {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_B; ++i)
      REQUIRE(VdB[i] == Approx(correct_VdB[i]));

    delete[] correct_VdB;
  }

  SECTION("initialize VdW") {
    nBlocks = (n_W + nThreads - 1) / nThreads;
    InitializeVdW<<<nBlocks, nThreads>>>(n_W, d_VdW);
    cudaMemcpy(VdW, d_VdW, n_W * sizeof(float), cudaMemcpyDeviceToHost);    
    float *correct_VdW = new float[n_W] {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_W; ++i)
      REQUIRE(VdW[i] == Approx(correct_VdW[i]));

    delete[] correct_VdW;
  }

  SECTION("initialize VdW and VdB") {
    InitializeMomentum(n_layers, d_VdW, d_VdB, W_index, B_index);
    cudaMemcpy(VdW, d_VdW, n_W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VdB, d_VdB, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    float *correct_VdW = new float[n_W] {0.0f, 0.0f, 0.0f, 0.0f,
                                         0.0f, 0.0f, 0.0f, 0.0f};
    float *correct_VdB = new float[n_B] {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < n_W; ++i)
      REQUIRE(VdW[i] == Approx(correct_VdW[i]));
    for (int i = 0; i < n_B; ++i)
      REQUIRE(VdB[i] == Approx(correct_VdB[i]));
  }

  cudaFree(d_VdW); cudaFree(d_VdB);
  delete[] W_index; delete[] B_index; delete[] VdW; delete[] VdB;
}

TEST_CASE("update momentum parameters", "[updateMomentum]") {
  int n_W {8};
  float *VdW = new float[n_W] {0.015f, 0.035f, -0.02f, 0.007f,
                               0.1f, 0.2f, -1.0f, 0.5f};
  float *W = new float[n_W] {1.2f, -1.8f, -0.7f, 0.14f, 0.3f, 0.85f, -0.56f, 0.34f};
  float *dW = new float[n_W] {0.179496767558757f, 0.0f, 0.115390779144915f, 0.0f,
                              0.230781558289831f, 0.0f, -0.409820425931346f, 0.0f};
  int n_B {3};
  float *VdB = new float[n_B] {0.12f, 0.08f, -0.5f};
  float *B = new float[n_B] {0.1f, 0.2f, 0.67f};
  float *dB = new float[n_B] {0.256423953655368f, 0.0f, -0.457899917241728f};

  float *d_VdW, *d_W, *d_dW, *d_VdB, *d_B, *d_dB;
  cudaMalloc(&d_dW, n_W * sizeof(float));
  cudaMalloc(&d_dB, n_B * sizeof(float));

  cudaMemcpy(d_dW, dW, n_W * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dB, dB, n_B * sizeof(float), cudaMemcpyHostToDevice);

  float beta {0.9f};
  float learn_rate {0.12f};
  int nBlocks {0};

  SECTION("update momentum B") {
    cudaMalloc(&d_VdB, n_B * sizeof(float));
    cudaMalloc(&d_B, n_B * sizeof(float));
    cudaMemcpy(d_VdB, VdB, n_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n_B * sizeof(float), cudaMemcpyHostToDevice);

    nBlocks = (n_B + nThreads - 1) / nThreads;
    UpdateMomentumB<<<nBlocks, nThreads>>>(n_B, d_VdB, d_B, d_dB,
                                           beta, learn_rate);
    cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_B = new float[n_B] {0.0839629125561356f,
                                       0.19136f,
                                       0.7294947990069007f};
    for (int i = 0; i < n_B; ++i)
      REQUIRE(B[i] == Approx(correct_B[i]));

    cudaFree(d_VdB); cudaFree(d_B);
    delete[] correct_B;
  }

  SECTION("update momentum W") {
    cudaMalloc(&d_VdW, n_W * sizeof(float));
    cudaMalloc(&d_W, n_W * sizeof(float));
    cudaMemcpy(d_VdW, VdW, n_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);

    nBlocks = (n_W + nThreads - 1) / nThreads;
    UpdateMomentumW<<<nBlocks, nThreads>>>(n_W, d_VdW, d_W, d_dW,
                                           beta, learn_rate);
    cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_W = new float[n_W] {1.196226038789295f, -1.80378f,
                                       -0.699224689349739f, 0.139244f,
                                       0.286430621300522f, 0.8284f,
                                       -0.447082154888824f, 0.286f};
    for (int i = 0; i < n_W; ++i)
      REQUIRE(W[i] == Approx(correct_W[i]));

    cudaFree(d_VdW); cudaFree(d_W);
    delete[] correct_W;
  }

  SECTION("update momentum W and B") {
    cudaMalloc(&d_VdB, n_B * sizeof(float));
    cudaMalloc(&d_B, n_B * sizeof(float));
    cudaMemcpy(d_VdB, VdB, n_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_VdW, n_W * sizeof(float));
    cudaMalloc(&d_W, n_W * sizeof(float));
    cudaMemcpy(d_VdW, VdW, n_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, n_W * sizeof(float), cudaMemcpyHostToDevice);

    UpdateMomentum(n_W, d_VdW, d_W, d_dW, n_B, d_VdB, d_B, d_dB,
                   beta, learn_rate);
    cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_B = new float[n_B] {0.0839629125561356f,
                                       0.19136f,
                                       0.7294947990069007f};
    float *correct_W = new float[n_W] {1.196226038789295f, -1.80378f,
                                       -0.699224689349739f, 0.139244f,
                                       0.286430621300522f, 0.8284f,
                                       -0.447082154888824f, 0.286f};
    for (int i = 0; i < n_B; ++i)
      REQUIRE(B[i] == Approx(correct_B[i]));
    for (int i = 0; i < n_W; ++i)
      REQUIRE(W[i] == Approx(correct_W[i]));

    cudaFree(d_VdB); cudaFree(d_B);
    cudaFree(d_VdW); cudaFree(d_W);
    delete[] correct_B; delete[] correct_W;
  }

  cudaFree(d_VdW); cudaFree(d_W); cudaFree(d_dW);
  cudaFree(d_VdB); cudaFree(d_B); cudaFree(d_dB);

  delete[] VdW; delete[] W; delete[] dW;
  delete[] VdB; delete[] B; delete[] dB;
}
