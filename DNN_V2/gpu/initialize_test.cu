#include <limits>
#include "catch2/catch.hpp"
#include "initialize.h"

extern const int nThreads {128};

TEST_CASE("initialize B", "[initB]") {
  int n {7};
  float *B = new float[n] {};
  float *d_B;
  cudaMalloc(&d_B, n * sizeof(float));

  int nBlocks = (n + nThreads - 1) / nThreads;
  InitializeB<<<nBlocks, nThreads>>>(n, d_B);
  cudaMemcpy(B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_B);

  float *correct_answer = new float[n] {};
  for (int i = 0; i < n; ++i)
    REQUIRE(B[i] == correct_answer[i]);

  delete[] B; delete[] correct_answer;
}

TEST_CASE("scale with He et al. (2015)", "[scaleHe]") {
  int n {5};
  float *W = new float[n] {-2.64e1, -2e-6, 0.0f, 1e-5, 3.27e3};
  int divisor {7};
  float *d_W;
  cudaMalloc(&d_W, n * sizeof(float));
  cudaMemcpy(d_W, W, n * sizeof(float), cudaMemcpyHostToDevice);

  int nBlocks = (n + nThreads -1) / nThreads;
  ScaleWithHe<<<nBlocks, nThreads>>>(n, d_W, divisor);
  cudaMemcpy(W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_W);

  float *correct_answer = new float[n] {-14.1113935729760f, -1.06904496764970e-6, 0.0f,
                                         5.34522483824849e-6, 1747.88852210726f};
  for (int i = 0; i < n; ++i)
    REQUIRE(W[i] == Approx(correct_answer[i]).epsilon(std::numeric_limits<float>::epsilon()));

  delete[] W; delete[] correct_answer; 
}

TEST_CASE("initialize W with array of even size", "[initWEven]") {
  int num_layers {2};
  int *layer_dims = new int[num_layers+1] {2, 3, 2};
  int *W_index = new int[num_layers+1] {0, 6, 12};
  int n = W_index[num_layers]; // n is even
  float *W = new float[n] {};
  float *d_W;
  cudaMalloc(&d_W, n * sizeof(float));

  SECTION("without He scaling") {
    bool he = false;
    InitializeW(num_layers, layer_dims, W_index, d_W, he);
    cudaMemcpy(W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_W);
    float *correct_answer = new float[n] {-0.872311f, -1.49021f, -1.03715f, 0.406382f,
                                          0.236329f, 0.703597f, 0.409391f, -1.23093f,
                                          -0.165716f, -1.10039f, 0.409605f, 2.66032f};

    for (int i = 0; i < n; ++i)
      REQUIRE(W[i] == Approx(correct_answer[i]));

    delete[] layer_dims; delete[] W_index;
    delete[] W; delete[] correct_answer;
  }

  SECTION("with He scaling") {
    bool he = true;
    InitializeW(num_layers, layer_dims, W_index, d_W, he);
    cudaMemcpy(W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_W);
    float *correct_answer = new float[n] {-0.872311f, -1.49021f, -1.03715f, 0.406382f,
                                          0.236329f, 0.703597f, 0.334266f, -1.00505f,
                                          -0.135307f, -0.89846f, 0.334441f, 2.17214f};

    for (int i = 0; i < n; ++i)
      REQUIRE(W[i] == Approx(correct_answer[i]));

    delete[] layer_dims; delete[] W_index;
    delete[] W; delete[] correct_answer;
  }
}

TEST_CASE("initialize W with array of odd size", "[initWOdd]") {
  int num_layers {2};
  int *layer_dims = new int[num_layers+1] {2, 3, 1};
  int *W_index = new int[num_layers+1] {0, 6, 9};
  int n = W_index[num_layers]; // n is odd
  float *W = new float[n] {};
  float *d_W;
  cudaMalloc(&d_W, n * sizeof(float));

  SECTION("without He scaling") {
    bool he = false;
    InitializeW(num_layers, layer_dims, W_index, d_W, he);
    cudaMemcpy(W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_W);
    float *correct_answer = new float[n] {-0.872311f, -1.49021f, -1.03715f, 0.406382f,
                                          0.236329f, 0.703597f, 0.409391f, -1.23093f,
                                          -0.165716f};

    for (int i = 0; i < n; ++i)
      REQUIRE(W[i] == Approx(correct_answer[i]));

    delete[] layer_dims; delete[] W_index;
    delete[] W; delete[] correct_answer;
  }

  SECTION("with He scaling") {
    bool he = true;
    InitializeW(num_layers, layer_dims, W_index, d_W, he);
    cudaMemcpy(W, d_W, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_W);
    float *correct_answer = new float[n] {-0.872311f, -1.49021f, -1.03715f, 0.406382f,
                                          0.236329f, 0.703597f, 0.334266f, -1.00505f,
                                          -0.135307f};

    for (int i = 0; i < n; ++i)
      REQUIRE(W[i] == Approx(correct_answer[i]));

    delete[] layer_dims; delete[] W_index;
    delete[] W; delete[] correct_answer;
  }
}

TEST_CASE("initialize W and B", "[initWB]") {
  int num_layers {2};
  int *layer_dims = new int[num_layers+1] {2, 3, 2};
  int *W_index = new int[num_layers+1] {0, 6, 12};
  int n_W = W_index[num_layers];
  float *W = new float[n_W] {};
  int *Z_index = new int[num_layers+1] {0, 3, 5};
  int n_B = Z_index[num_layers];
  float *B = new float[n_B] {};
  bool he = true;

  float *d_W, *d_B;
  cudaMalloc(&d_W, n_W * sizeof(float));
  cudaMalloc(&d_B, n_B * sizeof(float));

  InitializeParameters(num_layers, layer_dims, W_index, d_W, Z_index, d_B, he);
  cudaMemcpy(W, d_W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, d_B, n_B * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_W); cudaFree(d_B);

  float *correct_W = new float[n_W] {-0.872311f, -1.49021f, -1.03715f, 0.406382f,
                                     0.236329f, 0.703597f, 0.334266f, -1.00505f,
                                     -0.135307f, -0.89846f, 0.334441f, 2.17214f};
  for (int i = 0; i < n_W; ++i)
    REQUIRE(W[i] == Approx(correct_W[i]));

  float *correct_B = new float[n_B] {};
  for (int i = 0; i < n_B; ++i)
    REQUIRE(B[i] == Approx(correct_B[i]));

  delete[] layer_dims; delete[] W_index;
  delete[] W; delete[] Z_index;
  delete[] B; delete[] correct_W;
  delete[] correct_B;
}
