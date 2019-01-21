#include <curand.h>
#include <random> // for use in InitializeParametersSerial

extern const int nThreads;

// Initialize B to zero
__global__
void InitializeB(const int n, float *B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    B[i] = 0.0f;
}

// Scale W according to He et al. (2015)
__global__
void ScaleWithHe(const int n, float *W, const int divisor) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    W[i] *= sqrtf(2.0f / static_cast<float>(divisor));  
}

// Initialize W with random numbers with mean 0.0 and stddev 1.0
void InitializeW(const int num_layers, const int *layer_dims,
                 const int *W_index, float *W, const bool he) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
  curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST);
  curandSetPseudoRandomGeneratorSeed(gen, 1234);
  curandGenerateNormal(gen, W, W_index[num_layers], 0.0f, 1.0f);
  curandDestroyGenerator(gen);

  // If he is true, use scaling according to He et al. (2015)
  if (he) {
    int n {0};
    int nBlocks {0};
    for (int i = 1; i <= num_layers; ++i) {
      n = W_index[i] - W_index[i-1];
      nBlocks = (n + nThreads - 1) / nThreads;
      ScaleWithHe<<<nBlocks, nThreads>>>(n, W+W_index[i-1], layer_dims[i-1]);
    }
  }
}

void InitializeParameters(const int num_layers, const int *layer_dims,
                          const int *W_index, float *W, const int *Z_index,
                          float *B, const bool he) {
  int n {Z_index[num_layers]};
  int nBlocks = (n + nThreads - 1) / nThreads;
  InitializeB<<<nBlocks, nThreads>>>(n, B);

  InitializeW(num_layers, layer_dims, W_index, W, he);
}

// Main purpose of serial routine is comparison with/verification of cuda routine
// Note mt19937-generated numbers are different b/w cpu & gpu even with same seed
void InitializeParametersSerial(const int num_layers, const int *layer_dims,
                                const int *W_index, float *W, const int *Z_index,
                                float *B, const bool he) {
  // Initialize B to zero
  int n_B {Z_index[num_layers]};
  float *local_B = new float[n_B] {};
  cudaMemcpy(B, local_B, n_B * sizeof(float), cudaMemcpyHostToDevice);

  // Initialize W with random numbers with mean 0.0 and stddev 1.0
  int n_W {W_index[num_layers]};
  float *local_W = new float[n_W] {};
  std::mt19937 e2(1234);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  // If he is true, use scaling according to He et al. (2015)
  if (he) {
    int n {0};
    int start_index {0};
    for (int i = 1; i <= num_layers; ++i) {
      n = W_index[i] - W_index[i-1];
      start_index = W_index[i-1];
      for (int j = start_index; j < start_index+n; ++j) {
        local_W[j] = dist(e2) / sqrt(static_cast<float>(layer_dims[i-1]));
      }
    }
  }
  cudaMemcpy(W, local_W, n_W * sizeof(float), cudaMemcpyHostToDevice);

  delete[] local_B; delete[] local_W;
}
