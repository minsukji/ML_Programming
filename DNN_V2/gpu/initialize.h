#ifndef INITIALIZE_H
#define INITIALIZE_H

__global__
void InitializeB(const int n, float *B);

__global__
void ScaleWithHe(const int n, float *W, const int divisor);

void InitializeW(const int num_layers, const int *layer_dims,
                 const int *W_index, float *W, const bool he);

void InitializeParameters(const int num_layers, const int *layer_dims,
                          const int *W_index, float *W, const int *Z_index,
                          float *B, const bool he);

void InitializeParametersSerial(const int num_layers, const int *layer_dims,
                                const int *W_index, float *W, const int *Z_index,
                                float *B, const bool he);

#endif
