#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

__global__
void Sigmoid(const int n, const float *Z, float *A);

__global__
void Relu(const int n, const float *Z, float *A);

__global__
void SigmoidBackward(const int n, const float *dA,
                     const float *A, float *dZ);

__global__
void ReluBackward(const int n, const float *dA,
                  const float *Z, float *dZ);

#endif
