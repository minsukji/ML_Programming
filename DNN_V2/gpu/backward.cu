#include "cublas_v2.h"
#include "activation_func.h"
#include "backward.h"
#include "dropout.h"

extern const int nThreads;

__global__
void PartialCostPartialAL(const int n, const int *Y,
                          const float *A, float *dA) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dA[i] = -static_cast<float>(Y[i]) / A[i] +
            (1.0f - static_cast<float>(Y[i])) / (1.0f - A[i]);
}

void Backward(const int n_layers, const int *layer_dims, const int batch_size,
              const float *X, const int *Y, const float *W, const float *B,
              const float *Z, const float *A, float *dW, float *dB, float *dZ,
              float *dA, const int *W_index, const int *B_index, const int *Z_index,
              const float *oneVec, const float *layer_drop, const float *D,
              const float lambda) {
  int m, n, k, lda, ldb, ldc, n_elements;
  float alpha, beta;
  int W_curLoc, B_curLoc, Z_curLoc, Z_nextLoc;
  cublasHandle_t handle;
  cublasCreate(&handle);

  int l = n_layers;
  Z_curLoc = Z_index[l-1];
  n_elements = Z_index[l] - Z_index[l-1];
  int nBlocks = (n_elements + nThreads - 1) / nThreads;

  // Compute dA of the last layer
  PartialCostPartialAL<<<nBlocks, nThreads>>>(n_elements, Y,
                                              A+Z_curLoc, dA+Z_curLoc);

  // Backpropagation from the last layer L to the second layer
  for (l = n_layers; 1 < l; --l) {
    W_curLoc = W_index[l-1];
    B_curLoc = B_index[l-1];
    Z_curLoc = Z_index[l-1];
    Z_nextLoc = Z_index[l-2];
    n_elements = Z_index[l] - Z_index[l-1];
    nBlocks = (n_elements + nThreads - 1) / nThreads;

    // Compute dZ
    if (l == n_layers)
      SigmoidBackward<<<nBlocks, nThreads>>>(n_elements, dA+Z_curLoc,
                                             A+Z_curLoc, dZ+Z_curLoc);
    else
      ReluBackward<<<nBlocks, nThreads>>>(n_elements, dA+Z_curLoc,
                                          Z+Z_curLoc, dZ+Z_curLoc);

    // Compute dB
    m = layer_dims[l];
    n = batch_size;
    lda = m;
    alpha = 1.0f / batch_size;
    beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, dZ+Z_curLoc, lda,
                oneVec, 1, &beta, dB+B_curLoc, 1);

    // Compute dW
    m = layer_dims[l];
    n = layer_dims[l-1];
    k = batch_size;
    lda = m;
    ldb = n;
    ldc = m;
    if (lambda == 0.0f) {
      // dW without L2 regularization
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                  dZ+Z_curLoc, lda, A+Z_nextLoc, ldb, &beta, dW+W_curLoc, ldc);
    } else {
      // dW with L2 regularization
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                  dZ+Z_curLoc, lda, A+Z_nextLoc, ldb, &beta, dW+W_curLoc, ldc);

      alpha = 1.0f;
      beta = lambda / batch_size;
      lda = m;
      ldb = m;
      ldc = lda;
      cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha,
                  dW+W_curLoc, lda, &beta, W+W_curLoc, ldb, dW+W_curLoc, ldc);
    }

    // Compute dA 
    m = layer_dims[l-1];
    n = batch_size;
    k = layer_dims[l];
    lda = k;
    ldb = k;
    ldc = m;
    alpha = 1.0f;
    beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                W+W_curLoc, lda, dZ+Z_curLoc, ldb, &beta, dA+Z_nextLoc, ldc);
    if (D != nullptr && layer_drop[l-1] < 1.0f) {
      // Modify dA if dropout is applied
      float keep_prob = layer_drop[l-1];
      nBlocks = (m*n + nThreads - 1) / nThreads;
      ApplyDropout<<<nBlocks, nThreads>>>(m*n, D+Z_nextLoc, dA+Z_nextLoc, keep_prob);
    }
  }

  // Backpropagation for the first layer
  l = 1;
  W_curLoc = W_index[l-1];
  B_curLoc = B_index[l-1];
  Z_curLoc = Z_index[l-1];
  n_elements = Z_index[l] - Z_index[l-1];
  nBlocks = (n_elements + nThreads - 1) / nThreads;

  // Compute dZ
  ReluBackward<<<nBlocks, nThreads>>>(n_elements, dA+Z_curLoc,
                                      A+Z_curLoc, dZ+Z_curLoc);

  // Compute dB
  m = layer_dims[l];
  n = batch_size;
  lda = m;
  alpha = 1.0f / batch_size;
  beta = 0.0f;
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, dZ+Z_curLoc, lda,
              oneVec, 1, &beta, dB+B_curLoc, 1);

  // Compute dW. Use X instead of A
  m = layer_dims[l];
  n = layer_dims[l-1];
  k = batch_size;
  lda = m;
  ldb = n;
  ldc = m;
  if (lambda == 0.0f) {
    // dW without L2 regularization
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                dZ+Z_curLoc, lda, X, ldb, &beta, dW+W_curLoc, ldc);
  } else {
    // dW with L2 regularization
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
                dZ+Z_curLoc, lda, X, ldb, &beta, dW+W_curLoc, ldc);
    alpha = 1.0f;
    beta = lambda / batch_size;
    lda = m;
    ldb = m;
    ldc = lda;
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alpha,
                dW+W_curLoc, lda, &beta, W+W_curLoc, ldb, dW+W_curLoc, ldc);
  }
  cublasDestroy(handle);
}
