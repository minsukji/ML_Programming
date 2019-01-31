#include <curand.h>
#include "dropout.h"

bool CheckDropout(const int n_layers, float const *layer_dropout) {
  bool dropout {false};
  for (int l = 0; l < n_layers; ++l) {
    if (layer_dropout[l] < 1.0f) {
      dropout = true;
      break;
    }
  }
  return dropout;
}

void RandomlySelectDropout(const int n, float *D) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST);
  curandSetPseudoRandomGeneratorSeed(gen, 104);
  curandGenerateUniform(gen, D, n);
  curandDestroyGenerator(gen);
}

__global__
void ApplyDropout(const int n, const float *D, float *A,
                  const float keep_prob) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (D[i] > keep_prob)
      A[i] = 0.0f;
    else
      A[i] /= keep_prob;
  }
}
