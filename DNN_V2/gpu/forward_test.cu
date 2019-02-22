#include <limits>
#include "catch.hpp"
#include "forward.h"
#include "dropout.h"

extern const int nThreads {128};

TEST_CASE("forward propagation is computed", "[forwardProp]") {
  // Input layer (2); First hidden layer (4); Output layer (1)
  // W1(4,2), B1(4,1), W2(1,4), B2(1,1)
  // Batch size (2)
  int n_layers {2};
  int *layer_dims = new int[n_layers+1] {2, 4, 1};
  int batch_size {2};

  float *X = new float[2*batch_size] {0.7f, 0.45f, 0.9f, 0.1f};
  float *W = new float[8+4] {1.2f, -1.8f, -0.7f, 0.14f,
                             2.88f, 0.3f, 0.85f, -0.56f,
                             0.34f, 4.4f, -0.42f, -3.6f};
  float *B = new float[4+1] {0.1f, 0.2f, 0.67f, -0.3f, 0.15f};
  float *Z = new float[(4+1)*batch_size] {};
  float *A = new float[(4+1)*batch_size] {};

  float *d_X, *d_W, *d_B, *d_Z, *d_A;
  cudaMalloc(&d_X, (2*batch_size) * sizeof(float));
  cudaMalloc(&d_W, (8+4) * sizeof(float));
  cudaMalloc(&d_B, (4+1) * sizeof(float));
  cudaMalloc(&d_Z, ((4+1)*batch_size) * sizeof(float));
  cudaMalloc(&d_A, ((4+1)*batch_size) * sizeof(float));
  cudaMemcpy(d_X, X, (2*batch_size) * sizeof(float), cudaMemcpyHostToDevice);
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

  SECTION("without dropout") {
    float *layer_drop = new float[n_layers+1] {1.0f, 1.0f, 1.0f};

    float *d_D = nullptr;

    Forward(n_layers, layer_dims, batch_size, d_X, d_W, d_B, d_Z, d_A,
            W_index, B_index, Z_index, d_oneVec, layer_drop, d_D);
    cudaMemcpy(Z, d_Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(A, d_A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_Z
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, -0.454f, 1.468f,
                                     -1.39f, 0.125f, -0.23f, 0.67399f, 0.59662f};
    float *correct_A
      = new float[(4+1)*batch_size] {2.236f, 0.0f, 0.5625f, 0.0f, 1.468f,
                                     0.0f, 0.125f, 0.0f, 0.662396010418219f,
                                     0.644882635337264f};
    for (int i = 0; i < (4+1)*batch_size; ++i) {
      REQUIRE(Z[i] == Approx(correct_Z[i]));
      REQUIRE(A[i] == Approx(correct_A[i]));
    }
    delete[] layer_drop; delete[] correct_Z; delete[] correct_A;
  }

  SECTION("with dropout") {
    float *layer_drop = new float[n_layers+1] {1.0f, 0.7f, 1.0f};

    float *d_D;
    cudaMalloc(&d_D, ((4+1)*batch_size) * sizeof(float));
    RandomlySelectDropout((4+1)*batch_size, d_D);

    Forward(n_layers, layer_dims, batch_size, d_X, d_W, d_B, d_Z, d_A,
            W_index, B_index, Z_index, d_oneVec, layer_drop, d_D);
    cudaMemcpy(Z, d_Z, ((4+1)*batch_size) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(A, d_A, ((4+1)*batch_size) * sizeof(float), cudaMemcpyDeviceToHost);

    float *correct_Z
      = new float[(4+1)*batch_size] {2.236f, -0.925f, 0.5625f, -0.454f, 1.468f,
                                     -1.39f, 0.125f, -0.23f, 0.898557142857143f,
                                     0.788028571428572f};
    float *correct_A
      = new float[(4+1)*batch_size] {3.19428571428571f, 0.0f,
                                     0.803571428571429f, 0.0f,
                                     2.09714285714286f, 0.0f,
                                     0.178571428571429f, 0.0f,
                                     0.710652904814677f, 0.687407869971444f};
    for (int i = 0 ; i < (4+1)*batch_size; ++i) {
      REQUIRE(Z[i] == Approx(correct_Z[i]));
      REQUIRE(A[i] == Approx(correct_A[i]));
    }

    cudaFree(d_D);
    delete[] layer_drop; delete[] correct_Z; delete[] correct_A;
  }

  cudaFree(d_X); cudaFree(d_W); cudaFree(d_B);
  cudaFree(d_Z); cudaFree(d_A); cudaFree(d_oneVec);

  delete[] layer_dims; delete[] X; delete[] W; delete[] B; delete[] Z; delete[] A;
  delete[] W_index; delete[] B_index; delete[] Z_index; delete[] oneVec;
}
