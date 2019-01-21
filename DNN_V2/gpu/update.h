#ifndef UPDATE_H
#define UPDATE_H

__global__
void UpdateB(const int n_B, float *B, const float *dB, const float learn_rate);

__global__
void UpdateW(const int n_W, float *W, const float *dW, const float learn_rate);

void UpdateParameters(const int n_W, float *W, const float *dW,
                      const int n_B, float *B, const float *dB,
                      const float learn_rate);

#endif
