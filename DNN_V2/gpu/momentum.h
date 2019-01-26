#ifndef MOMENTUM_H
#define MOMENTUM_H

__global__
void InitializeVdB(const int n, float *VdB);

__global__
void InitializeVdW(const int n, float *VdW);

void InitializeMomentum(const int n_layers, float *VdW,
                        float *VdB, const int *W_index,
                        const int *B_index);

__global__
void UpdateMomentumB(const int n_B, float *VdB, float *B, const float *dB,
                     const float beta, const float learn_rate);

__global__
void UpdateMomentumW(const int n_W, float *VdW, float *W, const float *dW,
                     const float beta, const float learn_rate);

void UpdateMomentum(const int n_W, float *VdW, float *W, const float *dW,
                    const int n_B, float *VdB, float *B, const float *dB,
                    const float beta, const float learn_rate);

#endif
