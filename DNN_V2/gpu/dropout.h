#ifndef DROPOUT_H
#define DROPOUT_H

void RandomlySelectDropout(const int n, float *D);

__global__
void ApplyDropout(const int n, const float *D, float *A,
                  const float keep_prob);

#endif
