#include "cublas_v2.h"
#include "compute_cost.h"
#include <iostream>

extern const int nThreads;

__global__
void CrossEntropyElementWiseCompute(const int n, const float *AL,
                                    const int *Y, float *cost_temp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (Y[i] == 0)
      cost_temp[i] = -logf(1.0f - AL[i]);
    else if (Y[i] == 1)
      cost_temp[i] = -logf(AL[i]);
  }
}

void CostCrossEntropy(const int n, const float *AL, const int *Y,
                      float *cost_temp, float *cost) {
  int nBlocks = (n + nThreads - 1) / nThreads;
  CrossEntropyElementWiseCompute<<<nBlocks, nThreads>>>(n, AL, Y, cost_temp);
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSasum(handle, n, cost_temp, 1, cost);
  *cost /= static_cast<float>(n);
  cublasDestroy(handle);
}

void CostL2Regularization(const int batch_size, const int n_W, const float *W,
                          const float lambda, float *cost) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float result_init {0.0f};
  float *result = &result_init;
  cublasSdot(handle, n_W, W, 1, W, 1, result);
  //cublasSnrm2(handle, n_W, W, 1, result);
  //*result *= *result;
  *result *= 0.5f * lambda / static_cast<float>(batch_size);
  std::cout <<"L2 reg. cost: " << *result << '\n';
  *cost += *result;
  cublasDestroy(handle);
}

void ComputeCostBinaryClass(const int batch_size, const float *AL,
                            const int *Y, float *cost_temp, const float lambda,
                            const int n_W, const float *W, float *cost) {
  CostCrossEntropy(batch_size, AL, Y, cost_temp, cost);

  if (lambda != 0.0f) {
    CostL2Regularization(batch_size, n_W, W, lambda, cost);
  }
}

// Purpose of serial routine is i) comparison with/verification of gpu routine;
// ii) in case 'nan' or 'inf' occurs, serial routine is easier to debug
void ComputeCostBinaryClassSerial(const int batch_size, const float *AL,
                                  const int *Y, const float lambda,
                                  const int n_W, const float *W, float *cost) {
  int n {batch_size};
  float *local_AL = new float[n] {};
  int *local_Y = new int[n] {};
  cudaMemcpy(local_AL, AL, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(local_Y, Y, n * sizeof(float), cudaMemcpyDeviceToHost);

  float sum {0.0f};
  float temp {0.0f};
  for (int i = 0; i < n; ++i) {
    if (local_Y[i] == 0)
      temp = -std::log(1.0f - local_AL[i]);
    else if (local_Y[i] == 1)
      temp = -std::log(local_AL[i]);
    sum += temp;
  }
  *cost = sum / static_cast<float>(batch_size);

  if (lambda != 0.0f) {
    float *local_W = new float[n_W] {};
    cudaMemcpy(local_W, W, n_W * sizeof(float), cudaMemcpyDeviceToHost);
    sum = 0.0f;
    for (int i = 0; i < n_W; ++i)
      sum += local_W[i] * local_W[i];
    sum *= 0.5f * lambda / static_cast<float>(batch_size);
    *cost += sum;
    delete[] local_W;
  }

  delete[] local_AL; delete[] local_Y;
}
