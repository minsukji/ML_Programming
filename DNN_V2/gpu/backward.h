#ifndef BACKWARD_H
#define BACKWARD_H

__global__
void PartialCostPartialAL(const int n, const int *Y,
                          const float *A, float *dA);

void Backward(const int n_layers, const int *layer_dims, const int batch_size,
              const float *X, const int *Y, const float *W, const float *B,
              const float *Z, const float *A, float *dW, float *dB, float *dZ,
              float *dA, const int *W_index, const int *B_index, const int *Z_index,
              const float *oneVec, const float *layer_drop, const float *D,
              const float lambda);

#endif
