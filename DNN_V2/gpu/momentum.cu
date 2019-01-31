#include "momentum.h"

extern const int nThreads;

__global__
void InitializeVdB(const int n, float *VdB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    VdB[i] = 0.0f;
}

__global__
void InitializeVdW(const int n, float *VdW) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    VdW[i] = 0.0f;
}

void InitializeMomentum(const int n_layers, float *VdW,
                        float *VdB, const int *W_index,
                        const int *B_index) {
  int n = B_index[n_layers];
  int nBlocks = (n + nThreads - 1) / nThreads;
  InitializeVdB<<<nBlocks, nThreads>>>(n, VdB);

  n = W_index[n_layers];
  nBlocks = (n + nThreads - 1) / nThreads;
  InitializeVdW<<<nBlocks, nThreads>>>(n, VdW);
}

__global__
void UpdateMomentumB(const int n_B, float *VdB, float *B, const float *dB,
                     const float beta, const float learn_rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_B) {
    VdB[i] = beta * VdB[i] + (1.0f - beta) * dB[i];
    B[i] -= learn_rate * VdB[i];
  }
}

__global__
void UpdateMomentumW(const int n_W, float *VdW, float *W, const float *dW,
                     const float beta, const float learn_rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_W) {
    VdW[i] = beta * VdW[i] + (1.0f - beta) * dW[i];
    W[i] -= learn_rate * VdW[i];
  }
}

void UpdateMomentum(const int n_W, float *VdW, float *W, const float *dW,
                    const int n_B, float *VdB, float *B, const float *dB,
                    const float beta, const float learn_rate) {
  int nBlocks {0};
  nBlocks = (n_B + nThreads - 1) / nThreads;
  UpdateMomentumB<<<nBlocks, nThreads>>>(n_B, VdB, B, dB,
                                         beta, learn_rate);

  nBlocks = (n_W + nThreads - 1) / nThreads;
  UpdateMomentumW<<<nBlocks, nThreads>>>(n_W, VdW, W, dW,
                                         beta, learn_rate);
}
