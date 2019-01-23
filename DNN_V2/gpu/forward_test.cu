#include <limits>
#include "catch2/catch.hpp"
#include "forward.h"
#include "dropout.h"

extern const int nThreads {128};

void Forward(const int n_layers, const int *layer_dims, const int batch_size,
             const float *X, const float *W, const float *B, float *Z, float *A,
             const int *W_index, const int *B_index, const int *Z_index,
             const float *oneVec, const float *layer_drop, const float *D);

TEST_CASE("forward propagation is computed", "[forwardProp]") {
  // Input layer (2); First hidden layer (4); Output layer (1)
  // W1(4,2), B1(4,1), W2(1,4), B2(1,1)
  // Batch size (1)
  int n_layers {2};
  int *layer_dims = new int[n_layers+1] {2, 4, 1};
  int batch_sze {2};

  float *X = new float[2*batch_size] {0.7f, 0.45f, 0.9f, 0.1f};
  float *W = new float[8+4] {1.2f, -1.8f, -0.7f, 0.14f,
                             2.88f, 0.3f, 0.85f, -0.56f,
                             0.34f};
  float *B = new float[4+1] {};
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

  int *W_index = new int[n_layers+1] {};
  int *B_index = new int[n_layers+1] {};
  int *Z_index = new int[n_layers+1] {};

  float *oneVec = new float[batch_size] {};

  SECTION("without dropout") {
    float *layer_drop = new float[n_layers+1] {1.0f, 1.0f, 1.0f};

    float *d_D = nullptr;

    Forward(n_layers, layer_dims, batch_size, d_X, d_W, d_B, d_Z, d_A,
            W_index, B_index, Z_index, d_oneVec, layer_drop, d_D);
  }

  SECTION("with dropout") {
    float *layer_drop = new float[n_layers+1] {1.0f, 0.7f, 1.0f};

    float *d_D;
    cudaMalloc(&d_D, (5*batch_size) * sizeof(float));
    RandomlySelectDropout();

    Forward(n_layers, layer_dims, batch_size, d_X, d_W, d_B, d_Z, d_A,
            W_index, B_index, Z_index, d_oneVec, layer_drop, d_D);
  }
}
