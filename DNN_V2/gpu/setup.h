#ifndef SETUP_H
#define SETUP_H

__global__
void FillOneVector(const int n, float *oneVec);

void OneVector(const int n, float *oneVec);

void VectorIndex(const int n_layers, const int batch_size,
                 const int *layer_dims, int *W_index,
                 int *B_index, int *Z_index);

#endif
