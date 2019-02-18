#ifndef COMPUTE_COST_H
#define COMPUTE_COST_H

__global__
void CrossEntropyElementWiseCompute(const int n, const float *AL,
                                    const int *Y, float *cost_temp);

void CostCrossEntropy(const int n, const float *AL, const int *Y,
                      float *cost_temp, float *cost);

void CostL2Regularization(const int batch_size, const int n_W, const float *W,
                          const float lambda, float *cost);

void ComputeCostBinaryClass(const int batch_size, const float *AL,
                            const int *Y, float *cost_temp, const float lambda,
                            const int n_W, const float *W, float *cost);

void ComputeCostBinaryClassSerial(const int batch_size, const float *AL,
                                  const int *Y, const float lambda,
                                  const int n_W, const float *W, float *cost);

float ComputeAccuracyBinaryClass(const int n, const float *predict_prob,
                                 const int *out_bin, const float thres);

#endif
