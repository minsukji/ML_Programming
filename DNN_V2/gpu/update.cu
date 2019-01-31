#include "update.h"

extern const int nThreads;

__global__
void UpdateB(const int n_B, float *B, const float *dB, const float learn_rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_B)
    B[i] -= learn_rate * dB[i];
}

__global__
void UpdateW(const int n_W, float *W, const float *dW, const float learn_rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_W)
    W[i] -= learn_rate * dW[i];
}

// Update parameters W and B using gradients dW, dB and learning rate
// Not used if momentum optimizer is used (see momentum.cu instead)
// Not used if ADAM optimizer is used (see adam.cu instead)
void UpdateParameters(const int n_W, float *W, const float *dW,
                      const int n_B, float *B, const float *dB,
                      const float learn_rate) {
  int nBlocks {0};
  nBlocks = (n_B + nThreads - 1) / nThreads;
  UpdateB<<<nBlocks, nThreads>>>(n_B, B, dB, learn_rate);

  nBlocks = (n_W + nThreads - 1) / nThreads;
  UpdateW<<<nBlocks, nThreads>>>(n_W, W, dW, learn_rate);
}
