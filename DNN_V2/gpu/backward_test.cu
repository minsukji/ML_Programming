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
  float *B = new float[4+1] {0.1f, 0.2f, 0.67f, 0.3f, 0.15f};
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
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, 0.146f, 1.468f,
                                     -1.39f, 0.125f, 0.37f, 0.14839f, -0.73538f};
    float *A
      = new float[(4+1)*batch_size] {2.236f, 0.0f, 0.5625f, 0.146f, 1.468f,
                                     0.0f, 0.125f, 0.37f,
                                     0.537029576908463f, 0.324015235550852f};
    cudaMemcpy(d_Z, Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);

    Backward(n_layers, layer_dims, batch_size, d_X, d_Y, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index, d_oneVec,
             layer_drop, d_D, lambda);
    cudaMemcpy(dW, d_dW, (8+4) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, (4+1) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_dW
      = new float[8+4] {-0.158519149308613f, 0.0f, 0.195817772675345f, 1.678438051502957f,
                        -0.046908978362138f, 0.0f, 0.057946385035582f, 0.496683300304992f,
                        -1.013773750122013f, 0.0f, -0.172459479272566f, -0.158854022308775};
    float *correct_dB
      = new float[4+1] {-0.193622381881916f, 0.0f, 0.239180589383544f, 2.050119337573233f,
                        -0.569477593770343f};

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
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, 0.146f, 1.468f,
                                     -1.39f, 0.125f, 0.37f, 0.14839f, -0.73538f};
    float *A
      = new float[(4+1)*batch_size] {2.236f, 0.0f, 0.5625f, 0.146f, 1.468f,
                                     0.0f, 0.125f, 0.37f,
                                     0.537029576908463f, 0.324015235550852f};
    cudaMemcpy(d_Z, Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);

    Backward(n_layers, layer_dims, batch_size, d_X, d_Y, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index, d_oneVec,
             layer_drop, d_D, lambda);
    cudaMemcpy(dW, d_dW, (8+4) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, (4+1) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_dW
      = new float[8+4] {-0.0985191493086126f, -0.09f, 0.1608177726753449f, 1.6854380515029568f,
                        0.0970910216378619f, 0.015f, 0.1004463850355823f, 0.4686833003049916f,
                        -0.996773750122013f, 0.22f, -0.193459479272566f, -0.338854022308775f};
    float *correct_dB
      = new float[4+1] {-0.193622381881916f, 0.0f, 0.239180589383544f, 2.050119337573233f,
                        -0.569477593770343f};

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
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, 0.146f, 1.468f,
                                     -1.39f, 0.125f, 0.37f, 0.1477f, -1.114828571428572f};
    float *A
      = new float[(4+1)*batch_size] {3.194285714285715f, 0.0f,
                                     0.803571428571429f, 0.208571428571429f,
                                     2.097142857142857f, 0.0f,
                                     0.178571428571429f, 0.528571428571429f,
                                     0.536858018652687f, 0.246971789920003f}; 
    cudaMemcpy(d_Z, Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyHostToDevice);

    Backward(n_layers, layer_dims, batch_size, d_X, d_Y, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index, d_oneVec,
             layer_drop, d_D, lambda);
    cudaMemcpy(dW, d_dW, (8+4) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, (4+1) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_dW
      = new float[8+4] {-0.243324588460814f, 0.0f, 0.300577432804535f, 1.742722429042278f,
                        -0.068902630206328f, 0.0f, 0.085115013784287f, 0.193635825449142f,
                        -1.529307773350020f, 0.0f, -0.253318493405617f, -0.247313690718790f};
    float *correct_dB
      = new float[4+1] {-0.295355617918061f, 0.0f, 0.364851057428193f, 1.936358254491420f,
                        -0.608085095713655};

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
