#ifndef ADAM_H
#define ADAM_H

__global__
void InitializeVdBSdB(const int n, float *VdB, float *SdB);

__global__
void InitializeVdWSdW(const int n, float *VdW, float *SdW);

void InitializeAdam(const int n_layers, float *VdW, float *SdW,
                    float *VdB, float *SdB, const int *W_index,
                    const int *B_index);

__global__
void UpdateAdamB(const int n_B, float *VdB, float *SdB, float *B,
                 const float *dB, const float beta1, const float beta2,
                 const float learn_rate, const int t);

__global__
void UpdateAdamW(const int n_W, float *VdW, float *SdW, float *W,
                 const float *dW, const float beta1, const float beta2,
                 const float learn_rate, const int t);

void UpdateAdam(const int n_W, float *VdW, float *SdW, float *W, const float *dW,
                const int n_B, float *VdB, float *SdB, float *B, const float *dB,
                const float beta1, const float beta2, const float learn_rate,
                const int t);

#endif
