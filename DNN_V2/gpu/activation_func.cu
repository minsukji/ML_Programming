#include "activation_func.h"

// Compute Sigmoid function
__global__
void Sigmoid(const int n, const float *Z, float *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] = 1.0f / (1.0f + expf(-Z[i]));
}

// Compute Rectified-Linear-Unit function
__global__
void Relu(const int n, const float *Z, float *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] = fmaxf(Z[i], 0.0f);
}

// Compute derivative of Sigmoid function
// For use in Backprop, dA is multiplied to get dZ
__global__
void SigmoidBackward(const int n, const float *dA,
                     const float *A, float *dZ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dZ[i] = dA[i] * A[i] * (1.0f - A[i]);
}

// Compute derivative of Relu function
// For use in Backprop, dA is multiplied to get dZ
__global__
void ReluBackward(const int n, const float *dA,
                  const float *Z, float *dZ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (Z[i] > 0.0f)
      dZ[i] = dA[i];
    else
      dZ[i] = 0.0f;
  }
}
