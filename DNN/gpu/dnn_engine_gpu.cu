#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>
#include <curand.h>

constexpr int nThreads {1024};

// Apply sigmoid function element-wise to a matrix.
__global__
void SigmoidCuda(const float *Z, const int n, float *A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        A[i] = 1.0f / (1.0f + expf(-Z[i]));
}

// Apply rectified-linear-unit function element-wise to a matrix.
__global__
void ReluCuda(const float *Z, const int n, float *A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        A[i] = fmaxf(Z[i], 0.0f);
}

// Compute derivative of sigmoid function element-wise for a matrix.
// For a = sigmoid(z), derivative of a with respect to z is a*(1-a).
__global__
void SigmoidBackwardCuda(const int n, const float *dA, const float *A, float *dZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dZ[i] = dA[i] * (A[i] * (1.0f - A[i]));
}

// Compute derivative of relu function element-wise for a matrix.
// For a = relu(z), derivative of a with repsct to z is: 1 if z is positive, 0 otherwise.
__global__
void ReluBackwardCuda(const int n, const float *dA, const float *Z, float *dZ)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (Z[i] > 0.0f)
            dZ[i] = dA[i];
        else
            dZ[i] = 0.0f;
    }
}

__global__
void InitParams2Cuda(float *W, const int n, const int divisor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        W[i] /= static_cast<float>(divisor);
}


// Randomly initialize parameters (W matrix and b vector) of each layer of DNN.
// layer_dims contains the number of activations from input to output layer.
void InitParamsCuda(const int *layer_dims, const int num_layers, const int *W_index, float *W, float *B) {
    //curandStatus_t stat;
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_QUASI_DEFAULT);
    //stat = curandGenerateNormal(gen, W, W_index[num_layers], 0.0f, 1.0f);
    curandGenerateNormal(gen, W, W_index[num_layers], 0.0f, 1.0f);
    //if (stat == CURAND_STATUS_SUCCESS) std::cout << "Success" << '\n';
    //if (stat == CURAND_STATUS_NOT_INITIALIZED) std::cout << "Not initialized" << '\n';
    //if (stat == CURAND_STATUS_PREEXISTING_FAILURE) std::cout << "Preexisting failure" << '\n';
    //if (stat == CURAND_STATUS_LAUNCH_FAILURE) std::cout << "Launch failure" << '\n';
    //if (stat == CURAND_STATUS_LENGTH_NOT_MULTIPLE) std::cout << "Length not multiple" << '\n';
    //stat = curandGenerateNormal(gen, B, 33, 0.0f, 0.0f);
    //curandGenerateNormal(gen, B, 33, 0.0f, 0.0f);
    curandDestroyGenerator(gen);

    for (int i = 1; i <= num_layers; ++i) {
        int nBlocks = ((W_index[i] - W_index[i-1]) + nThreads - 1) / nThreads;
        InitParams2Cuda<<<nBlocks, nThreads>>>(W+W_index[i-1], W_index[i] - W_index[i-1], layer_dims[i-1]);
    }
}

// Carry out forward propagation.
// Arguments are matrix X (input data) and params.
void ForwardCuda(const float *X, const float *W, const float *B, const float *rowVec, const int *layer_dims, const int num_layers,
                 const int n_samples, const int *W_index, const int *B_index, const int *Z_index, float *Z, float *A) {
    
    int m, n, k, lda, ldb, ldc;
    float alf = 1.0f;
    float bet = 0.0f;
    const float *alpha = &alf;
    const float *beta = &bet;
    int incx = 1, incy = 1;
    int W_curLoc, B_curLoc, Z_curLoc, Z_lasLoc;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t stat;
    for (int l = 1; l <= num_layers; ++l) {
        W_curLoc = W_index[l-1];
        B_curLoc = B_index[l-1];
        Z_curLoc = Z_index[l-1];
        //std::cout << "W_curLoc: " << W_curLoc << ", B_curLoc: " << B_curLoc << ", Z_curLoc: " << Z_curLoc << '\n';

        m = layer_dims[l];
        n = n_samples;
        k = layer_dims[l-1];
        //std::cout << "m: " << m << ", n: " << n << ", k: " << k << '\n';
        lda = m;
        ldb = k;
        ldc = m;
       
        if (l == 1) 
            stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, W+W_curLoc, lda, X, ldb, beta, Z+Z_curLoc, ldc);
        else
            stat=cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, W+W_curLoc, lda, A+Z_lasLoc, ldb, beta, Z+Z_curLoc, ldc);
        if (stat != CUBLAS_STATUS_SUCCESS) std::cout << "PROBLEM" << '\n';

        stat=cublasSger(handle, m, n, alpha, B+B_curLoc, incx, rowVec, incy, Z+Z_curLoc, lda);
        if (stat != CUBLAS_STATUS_SUCCESS) std::cout << "PROBLEM" << '\n';

        int nBlocks = (m*n + nThreads - 1) / nThreads;
        if (l < num_layers)
            ReluCuda<<<nBlocks, nThreads>>>(Z+Z_curLoc, m*n, A+Z_curLoc); // Relu(W * X + B)
        else if (l == num_layers)
            SigmoidCuda<<<nBlocks, nThreads>>>(Z+Z_curLoc, m*n, A+Z_curLoc); // Sigmoid(W * X + B)

        Z_lasLoc = Z_curLoc;
    }
    cublasDestroy(handle);
}

__global__
void ComputeCostCuda2(const float *AL, const int *Y, const int n_samples, float *d_cost) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i == 0)
        *d_cost = 0.0f;

    __shared__ float temp[nThreads];
    if (i < n_samples)
        //temp[threadIdx.x] = static_cast<float>(Y[i]) * logf(AL[i]) + (1.0f - static_cast<float>(Y[i])) * logf(1.0f - AL[i]);
        temp[threadIdx.x] = static_cast<float>(Y[i]) * logf(AL[i]) + static_cast<float>((1 - Y[i])) * logf(1.0f - AL[i]);

    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        int index = (n_samples < nThreads) ? n_samples : nThreads;
        //for (int j = 0; j < nThreads; ++j)
        for (int j = 0; j < index; ++j)
            sum += temp[j];
        sum /= -static_cast<float>(n_samples);
        atomicAdd(d_cost, sum);
    }
}

// Given AL (output layer activation) and Y (), compute the cost function (cross entropy).
void ComputeCostCuda(const float *AL, const int *Y, const int n_samples, float *cost, float *d_cost)
{
    int nBlocks = (n_samples + nThreads - 1) / nThreads;
    ComputeCostCuda2<<<nBlocks, nThreads>>>(AL, Y, n_samples, d_cost);
    cudaMemcpy(cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
}

__global__
void BackwardCuda2(const int n, const int *Y, const float *A, float *dA) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        dA[i] = -static_cast<float>(Y[i]) / A[i] + (1.0f - static_cast<float>(Y[i])) / (1.0f - A[i]);
}
// Carry out backward propagation.
// Returns gradients of cost function with respect to W matrix and b vector
// In order to utilize the push_back method of vector data structure, grads
void BackwardCuda(const float *W, const float *B, const float *Z, const float *A, const float *X, const int *Y,
                  const float *oneVec, const int *layer_dims, const int num_layers, const int n_samples,
                  const int *W_index, const int *B_index, const int *Z_index, float *dW, float *dB, float *dZ, float *dA)
{
    int m, n, k, lda, ldb, ldc, n_elements;
    float alf1 = 1.0f;
    float alf2 = 1.0f / n_samples;
    float bet = 0.0f;
    const float *alpha1 = &alf1;
    const float *alpha2 = &alf2;
    const float *beta = &bet;
    int incx = 1, incy = 1;
    int W_curLoc, B_curLoc, Z_curLoc, Z_nexLoc;
    cublasHandle_t handle;
    cublasCreate(&handle);

    Z_curLoc = Z_index[num_layers-1];
    n_elements = Z_index[num_layers]-Z_index[num_layers-1];

    int nBlocks = (n_elements + nThreads - 1) / nThreads;
    BackwardCuda2<<<nBlocks, nThreads>>>(n_elements, Y, A+Z_curLoc, dA+Z_curLoc);

    int l;
    for (l = num_layers; l > 1; --l) {
        W_curLoc = W_index[l-1];
        B_curLoc = B_index[l-1];
        Z_curLoc = Z_index[l-1];
        Z_nexLoc = Z_index[l-2];
        n_elements = Z_index[l] - Z_index[l-1];

        nBlocks = (n_elements + nThreads - 1) / nThreads;
        if (l == num_layers)
            SigmoidBackwardCuda<<<nBlocks, nThreads>>>(n_elements, dA+Z_curLoc, A+Z_curLoc, dZ+Z_curLoc);
        else
            ReluBackwardCuda<<<nBlocks, nThreads>>>(n_elements, dA+Z_curLoc, Z+Z_curLoc, dZ+Z_curLoc);

        m = layer_dims[l];
        n = n_samples;
        lda = m;
        cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha2, dZ+Z_curLoc, lda, oneVec, incx, beta, dB+B_curLoc, incy);

        m = layer_dims[l];
        n = layer_dims[l-1];
        k = n_samples;
        lda = m;
        ldb = n;
        ldc = m;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha2, dZ+Z_curLoc, lda,  A+Z_nexLoc, ldb, beta, dW+W_curLoc, ldc);

        m = layer_dims[l-1];
        n = n_samples;
        k = layer_dims[l];
        lda = k;
        ldb = k;
        ldc = m;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha1, W+W_curLoc, lda, dZ+Z_curLoc, ldb, beta, dA+Z_nexLoc, ldc);
    }

    l = 1;
    W_curLoc = W_index[l-1];
    B_curLoc = B_index[l-1];
    Z_curLoc = Z_index[l-1];
    n_elements = Z_index[l] - Z_index[l-1];

    nBlocks = (n_elements + nThreads - 1) / nThreads;
    ReluBackwardCuda<<<nBlocks, nThreads>>>(n_elements, dA+Z_curLoc, A+Z_curLoc, dZ+Z_curLoc);

    m = layer_dims[l];
    n = n_samples;
    lda = m;
    cublasSgemv(handle, CUBLAS_OP_N, m, n, alpha2, dZ+Z_curLoc, lda, oneVec, incx, beta, dB+B_curLoc, incy);

    m = layer_dims[l];
    n = layer_dims[l-1];
    k = n_samples;
    lda = m;
    ldb = n;
    ldc = m;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha2, dZ+Z_curLoc, lda,  X, ldb, beta, dW+W_curLoc, ldc);

    cublasDestroy(handle);
}

__global__
void UpdateParams2Cuda(const float learning_rate, const int n, const float *dW, float *W) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        W[i] -= learning_rate * dW[i];
    
}

__global__
void UpdateParams3Cuda(const float learning_rate, const int n, const float *dB, float *B) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        B[i] -= learning_rate * dB[i];
    
}

// Update parameters
void UpdateParamsCuda(const float learning_rate, const int n_W, const int n_B, const float *dW, const float *dB, float *W, float *B)
{
    int nBlocks = (n_W + nThreads - 1) / nThreads;
    UpdateParams2Cuda<<<nBlocks, nThreads>>>(learning_rate, n_W, dW, W);

    nBlocks = (n_B + nThreads - 1) / nThreads;
    UpdateParams3Cuda<<<nBlocks, nThreads>>>(learning_rate, n_B, dB, B);
}

void VectorStartIndex(const int *layer_dims, int num_layers, int m,
                      int *W_index, int *B_index, int *Z_index) {
    int W_sum {0}, B_sum {0}, Z_sum {0};

    for (int i = 0; i < num_layers; ++i) {
        W_index[i] = W_sum;
        W_sum += layer_dims[i+1] * layer_dims[i];
        B_index[i] = B_sum;
        B_sum += layer_dims[i+1];
        Z_index[i] = Z_sum;
        Z_sum += layer_dims[i+1] * m;
    }
    W_index[num_layers] = W_sum;
    B_index[num_layers] = B_sum;
    Z_index[num_layers] = Z_sum;
}
