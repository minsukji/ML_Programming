#include "adam.h"

extern const int nThreads;

__global__
void InitializeVdBSdB(const int n, float *VdB, float *SdB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    VdB[i] = 0.0f;
    SdB[i] = 0.0f;
  }
}

__global__
void InitializeVdWSdW(const int n, float *VdW, float *SdW) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    VdW[i] = 0.0f;
    SdW[i] = 0.0f;
  }
}

void InitializeAdam(const int n_layers, float *VdW, float *SdW,
                    float *VdB, float *SdB, const int *W_index,
                    const int *B_index) {
  int n = B_index[n_layers];
  int nBlocks = (n + nThreads - 1) / nThreads;
  InitializeVdBSdB<<<nBlocks, nThreads>>>(n, VdB, SdB);

  n = W_index[n_layers];
  nBlocks = (n + nThreads - 1) / nThreads;
  InitializeVdWSdW<<<nBlocks, nThreads>>>(n, VdW, SdW);
}

__global__
void UpdateAdamB(const int n_B, float *VdB, float *SdB, float *B,
                 const float *dB, const float beta1, const float beta2,
                 const float learn_rate, const int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float epsilon {1e-8};
  float VdB_corrected, SdB_corrected;
  if (i < n_B) {
    VdB[i] = beta1 * VdB[i] + (1.0f - beta1) * dB[i];
    VdB_corrected = VdB[i] / (1.0f - powf(beta1, static_cast<float>(t)));
    SdB[i] = beta2 * SdB[i] + (1.0f - beta2) * dB[i] * dB[i];
    SdB_corrected = SdB[i] / (1.0f - powf(beta2, static_cast<float>(t)));
    B[i] -= learn_rate * VdB_corrected / (sqrtf(SdB_corrected) + epsilon);
  }
}

__global__
void UpdateAdamW(const int n_W, float *VdW, float *SdW, float *W,
                 const float *dW, const float beta1, const float beta2,
                 const float learn_rate, const int t) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float epsilon {1e-8};
  float VdW_corrected, SdW_corrected;
  if (i < n_W) {
    VdW[i] = beta1 * VdW[i] + (1.0f - beta1) * dW[i];
    VdW_corrected = VdW[i] / (1.0f - powf(beta1, static_cast<float>(t)));
    SdW[i] = beta2 * SdW[i] + (1.0f - beta2) * dW[i] * dW[i];
    SdW_corrected = SdW[i] / (1.0f - powf(beta2, static_cast<float>(t)));
    W[i] -= learn_rate * VdW_corrected / (sqrtf(SdW_corrected) + epsilon);
  }

}

void UpdateAdam(const int n_W, float *VdW, float *SdW, float *W, const float *dW,
                const int n_B, float *VdB, float *SdB, float *B, const float *dB,
                const float beta1, const float beta2, const float learn_rate,
                const int t) {
  int nBlocks {0};
  nBlocks = (n_B + nThreads - 1) / nThreads;
  UpdateAdamB<<<nBlocks, nThreads>>>(n_B, VdB, SdB, B, dB, beta1,
                                     beta2, learn_rate, t);

  nBlocks = (n_W + nThreads - 1) / nThreads;
  UpdateAdamW<<<nBlocks, nThreads>>>(n_W, VdW, SdW, W, dW, beta1,
                                     beta2, learn_rate, t);
}
