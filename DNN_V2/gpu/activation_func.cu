//-------1---------2---------3---------4---------5---------6---------7---------8
__global__
void Sigmoid(const int n, const float *Z, float *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] = 1.0f / (1.0f + expf(-Z[i]));
}

__global__
void Relu(const int n, const float *Z, float *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    A[i] = fmaxf(Z[i], 0.0f);
}
