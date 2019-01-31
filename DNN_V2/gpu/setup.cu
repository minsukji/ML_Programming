#include "setup.h"

extern const int nThreads;

__global__
void FillOneVector(const int n, float *oneVec) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    oneVec[i] = 1.0f;
}

void OneVector(const int n, float *oneVec) {
  int nBlocks = (n + nThreads - 1) / nThreads;
  FillOneVector<<<nBlocks, nThreads>>>(n, oneVec);
}

void VectorIndex(const int n_layers, const int batch_size,
                 const int *layer_dims, int *W_index,
                 int *B_index, int *Z_index) {
  int W_sum {0}, B_sum {0}, Z_sum {0};

  for (int l = 0; l < n_layers; ++l) {
    W_index[l] = W_sum;
    W_sum += layer_dims[l+1] * layer_dims[l];
    B_index[l] = B_sum;
    B_sum += layer_dims[l+1];
    Z_index[l] = Z_sum;
    Z_sum += layer_dims[l+1] * batch_size;
  }
  W_index[n_layers] = W_sum;  
  B_index[n_layers] = B_sum;
  Z_index[n_layers] = Z_sum;
}
