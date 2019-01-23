#include "cublas_v2.h"
#include "activation_func.h"
#include "dropout.h"

extern const int nThreads;

void Forward(const int n_layers, const int *layer_dims, const int batch_size,
             const float *X, const float *W, const float *B, float *Z, float *A,
             const int *W_index, const int *B_index, const int *Z_index,
             const float *oneVec, const float *layer_drop, const float *D) {
  int m, n, k, lda, ldb, ldc;
  int W_curLoc, B_curLoc, Z_curLoc, Z_lastLoc;

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Perform forward propagation from layer 1 to output layer L
  for (int l = 1; l <= n_layers; ++l) {
    W_curLoc = W_index[l-1];
    B_curLoc = B_index[l-1];
    Z_curLoc = Z_index[l-1];

    m = layer_dims[l];
    n = batch_size;
    k = layer_dims[l-1];
    lda = m;
    ldb = k;
    ldc = m;

    // 1 Compute Z (linear)
    // 1.1 W[l] * A[l-1]
    if (l == 1)
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0f,
                  W+W_curLoc, lda, X, ldb, 0.0f, Z+Z_curLoc, ldc);
    else
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0f,
                  W+W_curLoc, lda, A+Z_lastLoc, ldb, 0.0f, Z+Z_curLoc, ldc);
    // 1.2 W[l] * A[l-1] + B[l]
    cublasSger(handle, m, n, 1.0f, B+B_curLoc, 1, oneVec, 1, Z+Z_curLoc, lda);

    // 2 Compute A (non-linear)
    int nBlocks = (m*n + nThreads - 1) / nThreads;
    if (l < n_layers)
      Relu<<<nBlocks, nThreads>>>(m*n, Z+Z_curLoc, A+Z_curLoc);
    else if (l == n_layers)
      Sigmoid<<<nBlocks, nThreads>>>(m*n, Z+Z_curLoc, A+Z_curLoc);

    // 3 Modify A if dropout is applied
    if (D != nullptr) {
      float keep_prob = layer_drop[l];
      int nBlocks = (m*n + nThreads - 1) / nThreads;
      ApplyDropout<<<nBlocks, nThreads>>>(m*n, D+Z_curLoc, A+Z_curLoc, keep_prob);
    }

    Z_lastLoc = Z_curLoc;
  }

  cublasDestroy(handle);
}
