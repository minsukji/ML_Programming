#ifndef FORWARD_H
#define FORWARD_H

void Forward(const int n_layers, const int *layer_dims, const int batch_size,
             const float *X, const float *W, const float *B, float *Z, float *A,
             const int *W_index, const int *B_index, const int *Z_index,
             const float *oneVec, const float *layer_drop, const float *D);

#endif
