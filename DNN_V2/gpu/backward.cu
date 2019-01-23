#include "cublas_v2.h"

extern const int nThreads;

void Backward() {
  cublasHandle_t handle;
  cublasCreate(&handle);

  for (int l = n_layers; l > 1; --l) {

  }
}
