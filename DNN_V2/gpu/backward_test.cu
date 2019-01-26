#include <limits>
#include "catch2/catch.hpp"
#include "backward.h"
#include "dropout.h"

extern const int nThreads {128};

TEST_CASE("backward propagation is computed", "[backwardProp]") {
  // Input layer (2); First hidden layer (4); Output layer (1)
  // W1(4,2), B1(4,1), W2(1,4), B2(1,1)
  // Batch size (2)
  int n_layers {2};
  int *layer_dims = new int[n_layers+1] {2, 4, 1};
  int batch_size {2};

  float *X = new float[2*batch_size] {0.7f, 0.45f, 0.9f, 0.1f};
  int *Y = new int[batch_size] {1, 1};
  float *W = new float[8+4] {1.2f, -1.8f, -0.7f, 0.14f,
                             2.88f, 0.3f, 0.85f, -0.56f,
                             0.34f, 4.4f, -0.42f, -3.6f};
  float *B = new float[4+1] {0.1f, 0.2f, 0.67f, -0.3f, 0.15f};
  //float *B = new float[4+1] {0.1f, 0.2f, 0.67f, 0.3f, 0.15f};
  float *dW = new float[8+4] {};
  float *dB = new float[4+1] {};
  float *d_X, *d_W, *d_B, *d_Z, *d_A, *d_dW, *d_dB, *d_dZ, *d_dA;
  int *d_Y;
  cudaMalloc(&d_X, (2*batch_size) * sizeof(float));
  cudaMalloc(&d_Y, batch_size * sizeof(int));
  cudaMalloc(&d_W, (8+4) * sizeof(float));
  cudaMalloc(&d_B, (4+1) * sizeof(float));
  cudaMalloc(&d_Z, ((4+1)*batch_size) * sizeof(float));
  cudaMalloc(&d_A, ((4+1)*batch_size) * sizeof(float));
  cudaMalloc(&d_dW, (8+4) * sizeof(float));
  cudaMalloc(&d_dB, (4+1) * sizeof(float));
  cudaMalloc(&d_dZ, ((4+1)*batch_size) * sizeof(float));
  cudaMalloc(&d_dA, ((4+1)*batch_size) * sizeof(float));
  cudaMemcpy(d_X, X, (2*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, Y, batch_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, W, (8+4) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, (4+1) * sizeof(float), cudaMemcpyHostToDevice);

  int *W_index = new int[n_layers+1] {0, 8, 12};
  int *B_index = new int[n_layers+1] {0, 4, 5};
  int *Z_index = new int[n_layers+1] {0, 8, 10};

  float *oneVec = new float[batch_size] {};
  std::fill(oneVec, oneVec+batch_size, 1.0f);
  float *d_oneVec;
  cudaMalloc(&d_oneVec, batch_size * sizeof(float));
  cudaMemcpy(d_oneVec, oneVec, batch_size * sizeof(float), cudaMemcpyHostToDevice);

  SECTION("no regularization") {
    float lambda = 0.0f;
    float *layer_drop = new float[n_layers+1] {1.0f, 1.0f, 1.0f};
    float *d_D = nullptr;
    float *Z
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, -0.454f, 1.468f,
                                     -1.39f, 0.125f, -0.23f, 0.67399f, 0.59662f};
    float *A
      = new float[(4+1)*batch_size] {2.236f, 0.0f, 0.5625f, 0.0f, 1.468f,
                                     0.0f, 0.125f, 0.0f, 0.662396010418219f,
                                     0.644882635337264f};
    cudaMemcpy(d_Z, Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);

    Backward(n_layers, layer_dims, batch_size, d_X, d_Y, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index, d_oneVec,
             layer_drop, d_D, lambda);
    cudaMemcpy(dW, d_dW, (8+4) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, (4+1) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_dW
      = new float[8+4] {-0.094507831553631f, 0.0f, 0.116744968389779f, 0.0f,
                        -0.031863700402273f, 0.0f, 0.039361041673396f, 0.0f,
                        -0.638097406014880f, 0.0f, -0.117145957361297f, 0.0f};
    float *correct_dB
      = new float[4+1] {-0.117762630221568f, 0.0f, 0.145471484391349f, 0.0f, -0.346360677122259f};

    for (int i = 0; i < (8+4); ++i)
      REQUIRE(dW[i] == Approx(correct_dW[i]));
    for (int i = 0; i < (4+1); ++i)
      REQUIRE(dB[i] == Approx(correct_dB[i]));

    delete[] layer_drop; delete[] Z; delete[] A; delete[] correct_dW; delete[] correct_dB;
  }

  SECTION("with L2 regularization") {
    float lambda = 0.1f;
    float *layer_drop = new float[n_layers+1] {1.0f, 1.0f, 1.0f};
    float *d_D = nullptr;
    float *Z
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, -0.454f, 1.468f,
                                     -1.39f, 0.125f, -0.23f, 0.67399f, 0.59662f};
    float *A
      = new float[(4+1)*batch_size] {2.236f, 0.0f, 0.5625f, 0.0f, 1.468f,
                                     0.0f, 0.125f, 0.0f, 0.662396010418219f,
                                     0.644882635337264f};
    cudaMemcpy(d_Z, Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);

    Backward(n_layers, layer_dims, batch_size, d_X, d_Y, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index, d_oneVec,
             layer_drop, d_D, lambda);
    cudaMemcpy(dW, d_dW, (8+4) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, (4+1) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_dW
      = new float[8+4] {-0.03450783155363056f, -0.09f, 0.08174496838977893f, 0.007f,
                        0.11213629959772721f, 0.015f, 0.08186104167339578f, -0.028f,
                        -0.621097406014880f, 0.22f, -0.138145957361297f, -0.18f};
    float *correct_dB
      = new float[4+1] {-0.117762630221568f, 0.0f, 0.145471484391349f, 0.0f, -0.346360677122259f};

    for (int i = 0; i < (8+4); ++i)
      REQUIRE(dW[i] == Approx(correct_dW[i]));
    for (int i = 0; i < (4+1); ++i)
      REQUIRE(dB[i] == Approx(correct_dB[i]));

    delete[] layer_drop; delete[] Z; delete[] A; delete[] correct_dW; delete[] correct_dB;
  }

  SECTION("with dropout regularization") {
    float lambda = 0.0f;
    float *layer_drop = new float[n_layers+1] {1.0f, 0.7f, 1.0f};
    float *d_D;
    cudaMalloc(&d_D, ((4+1)*batch_size) * sizeof(float));
    RandomlySelectDropout((4+1)*batch_size, d_D);
    // with seed 104:
    // 0.147921, 0.460329, 0.30005, 0.907808, 0.0552578,
    // 0.0109106, 0.251024, 0.263781, 0.391988, 0.72534
    float *Z
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, -0.454f, 1.468f,
                                     -1.39f, 0.125f, -0.23f, 0.898557142857143f,
                                     0.788028571428572f};
    float *A
      = new float[(4+1)*batch_size] {3.19428571428571f, 0.0f,
                                     0.803571428571429f, 0.0f,
                                     2.09714285714286f, 0.0f,
                                     0.178571428571429f, 0.0f,
                                     0.710652904814677f, 0.687407869971444f};
    cudaMemcpy(d_Z, Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);

    Backward(n_layers, layer_dims, batch_size, d_X, d_Y, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index, d_oneVec,
             layer_drop, d_D, lambda);
    cudaMemcpy(dW, d_dW, (8+4) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, (4+1) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_dW
      = new float[8+4] {-0.117512714602032f, 0.0f, 0.145162765096628f, 0.0f,
                        -0.039213027131661f, 0.0f, 0.048439621750875f, 0.0f,
                        -0.789903822654501f, 0.0f, -0.144165540925224, 0.0f};
    float *correct_dB
      = new float[4+1] {-0.146185240409085f, 0.0f, 0.180581767564164f, 0.0f, -0.300969612606939f};

    for (int i = 0; i < (8+4); ++i)
      REQUIRE(dW[i] == Approx(correct_dW[i]));
    for (int i = 0; i < (4+1); ++i)
      REQUIRE(dB[i] == Approx(correct_dB[i]));

    cudaFree(d_D);
    delete[] layer_drop; delete[] Z; delete[] A; delete[] correct_dW; delete[] correct_dB;
  }

  cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); cudaFree(d_B); cudaFree(d_Z); cudaFree(d_A);
  cudaFree(d_dW); cudaFree(d_dB); cudaFree(d_dZ); cudaFree(d_dA); cudaFree(d_oneVec);
  delete[] layer_dims; delete[] X; delete[] W; delete[] B; delete[] dW; delete[] dB;
  delete[] W_index; delete[] B_index; delete[] Z_index; delete[] oneVec; 
}
