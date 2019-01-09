#ifndef DNN_ENGINE_GPU_H
#define DNN_ENGINE_GPU_H

__global__ void SigmoidCuda(const float *Z, const int n, float *A);
__global__ void ReluCuda(const float *Z, const int n, float *A);
void VectorStartIndex(const int *layer_dims, int num_layers, int m, int *W_index, int *B_index, int *Z_index);
void ForwardCuda(const float *X, const float *W, const float *B, const float *row_Vec, const int *layer_dims, const int num_layers,
                 const int n_samples, const int *W_index, const int *B_index, const int *Z_index, float *Z, float *A);
void InitParamsCuda(const int *layer_dims, const int num_layers, const int *W_index, float *W, float *B);
__global__ void InitParams2Cuda(float *W, const int n, const int divisor);
float ComputeCostCuda(const float *AL, const int *Y, const int n_samples, float *cost, float *d_cost);
void BackwardCuda(const float *W, const float *B, const float *Z, const float *A, const float *X, const int *Y,
                  const float *oneVec, const int *layer_dims, const int num_layers, const int n_samples,
                  const int *W_index, const int *B_index, const int *Z_index, float *dW, float *dB, float *dZ, float *dA);
void UpdateParamsCuda(const float learning_rate, const int n_W, const int n_B, const float *dW, const float *dB, float *W, float *B);

#endif
