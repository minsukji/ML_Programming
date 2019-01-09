#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>
#include "dnn_engine_gpu.h"
#include <cassert>
#include <cstdlib>

using std::isless;
using std::abs;

void TestSigmoidCuda() {
    float *h_Z = new float[6] {.0f, -.001f, -1.23f, 2.6f, .74f, .0002f};
    float *h_A = new float[6] ();
    float *d_Z, *d_A;

    cudaMalloc(&d_Z, 6 * sizeof(float));
    cudaMemcpy(d_Z, h_Z, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_A, 6 * sizeof(float));

    SigmoidCuda<<<1, 128>>>(d_Z, 6, d_A);
    cudaMemcpy(h_A, d_A, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    float *expected = new float[6] {.5f, .49975f, .22618142573f, .930861579657f, .67699585624f, .50005f};
    float epsilon = 1.0e-7;
    for (int i = 0; i < 6; ++i)
        if (!isless(abs(h_A[i]-expected[i]), epsilon))
            throw "TestSigmoidCuda failed";

    cudaFree(d_Z);
    cudaFree(d_A);
    delete[] h_Z;
    delete[] h_A;
    delete[] expected;
}

void TestReluCuda() {
    float *h_Z = new float[6] {.0f, -.001f, -1.23f, 2.6f, .74f, .0002f};
    float *h_A = new float[6] ();
    float *d_Z, *d_A;

    cudaMalloc(&d_Z, 6 * sizeof(float));
    cudaMemcpy(d_Z, h_Z, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_A, 6 * sizeof(float));

    ReluCuda<<<1, 128>>>(d_Z, 6, d_A);
    cudaMemcpy(h_A, d_A, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    float *expected = new float[6] {.0f, .0f, .0f, 2.6f, .74f, .0002f};
    float epsilon = 1.0e-7;
    for (int i = 0; i < 6; ++i)
        if (!isless(abs(h_A[i]-expected[i]), epsilon))
            throw "TestReluCuda failed";

    cudaFree(d_Z);
    cudaFree(d_A);
    delete[] h_Z;
    delete[] h_A;
    delete[] expected;
}

void TestVectorStartIndex() {
    int *layer_dims = new int[5] {12288, 20, 7, 5, 1};
    int num_layers {4};
    int m {11};
    int *W_index = new int[5]();
    int *B_index = new int[5]();
    int *Z_index = new int[5]();

    VectorStartIndex(layer_dims, num_layers, m, W_index, B_index, Z_index);

    int *W_expected = new int[5] {0, 245760, 245900, 245935, 245940};
    int *B_expected = new int[5] {0, 20, 27, 32, 33};
    int *Z_expected = new int[5] {0, 220, 297, 352, 363};

    for (int i = 0; i < 5; ++i)
        if (W_index[i] != W_expected[i])
            throw "TestVectorStartIndex failed";
    for (int i = 0; i < 5; ++i)
        if (B_index[i] != B_expected[i])
            throw "TestVectorStartIndex failed";
    for (int i = 0; i < 5; ++i)
        if (Z_index[i] != Z_expected[i])
            throw "TestVectorStartIndex failed";
}

void TestForwardCuda() {
    // Input layer (3); First hidden layer (2); Output layer (1)
    // W1(2,3), B1(2,1), W2(1,2), B2(1,1)
    // Number of samples (1)
    int *layer_dims = new int[3] {3, 2, 1};
    int num_layers {2};
    int n_samples {1};

    float *X = new float[3] {0.7f, 0.45f, 0.9f};
    float *W = new float[8] {1.2f, -1.8f, -0.7f, 0.14f, 0.3f, 0.85f, -0.56f, 0.34f};
    float *B = new float[3] {0.1f, 0.2f, 0.67f};
    float *rowVec = new float[n_samples]{};
    for (int i = 0; i < n_samples; ++i)
        rowVec[i] = 1.0f;

    float *d_X, *d_W, *d_B, *d_rowVec;
    cudaMalloc(&d_X, 3*sizeof(float));
    cudaMalloc(&d_W, 8*sizeof(float));
    cudaMalloc(&d_B, 3*sizeof(float));
    cudaMalloc(&d_rowVec, n_samples*sizeof(float));
    cudaMemcpy(d_X, X, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowVec, rowVec, n_samples*sizeof(float), cudaMemcpyHostToDevice);
    
    int *W_index = new int[3] {0, 6, 8};
    int *B_index = new int[3] {0, 2, 3};
    int *Z_index = new int[3] {0, 2, 3};

    float *d_Z, *d_A;
    cudaMalloc(&d_Z, 3*sizeof(float));
    cudaMalloc(&d_A, 3*sizeof(float));
    float *Z = new float[3]();
    float *A = new float[3]();

    ForwardCuda(d_X, d_W, d_B, d_rowVec, layer_dims, num_layers, n_samples, W_index, B_index, Z_index, d_Z, d_A);

    cudaMemcpy(Z, d_Z, 3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(A, d_A, 3*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_B);
    cudaFree(d_rowVec);
    cudaFree(d_Z);
    cudaFree(d_A); 

    float *Z_expected = new float[3] {0.895f, -0.232f, 0.1688f};
    float *A_expected = new float[3] {0.895f, 0.0f, 0.542100082758272f};

    float epsilon = 1.0e-7;
    for (int i = 0; i < 3; ++i) {
        //std::cout << "Z[" << i << "]: " << Z[i] << '\n';
        if (!isless(abs(Z[i]-Z_expected[i]), epsilon))
            throw "TestForwardCuda failed";
    }
    //std::cout << "===================================" << '\n';
    for (int i = 0; i < 3; ++i) {
        //std::cout << "A[" << i << "]: " << A[i] << '\n';
        if (!isless(abs(A[i]-A_expected[i]), epsilon))
            throw "TestForwardCuda failed";
    }

    delete[] X;
    delete[] W;
    delete[] B;
    delete[] rowVec;
    delete[] Z;
    delete[] A;
}

void TestInitParamsCuda() {
    int *layer_dims = new int[4] {3, 4, 8, 1};
    int num_layers {3};
    int *W_index = new int[4] {0, 12, 44, 52};
    float *W = new float[52] {};
    float *B = new float[13] {};
    float *d_W, *d_B;
    cudaMalloc(&d_W, 52*sizeof(float));
    cudaMalloc(&d_B, 13*sizeof(float));
    InitParamsCuda(layer_dims, num_layers, W_index, d_W, d_B);
    cudaMemcpy(W, d_W, 52*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, 13*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_W);
    cudaFree(d_B);
/*
    for (int i = 0; i < 52; ++i)
        std::cout << "i: " << i << ", W: " << W[i] << '\n';
    for (int i = 0; i < 13; ++i)
        std::cout << "i: " << i << ", B: " << B[i] << '\n';
*/
    delete[] W;
    delete[] B;
}

void TestComputeCostCuda() {
    int n_samples = 5;
    float *AL = new float[n_samples] {0.1f, 0.97f, 0.51f, 0.3f, 0.02f};
    float *cost = new float {0};
    int *Y = new int[n_samples] {0, 1, 1, 1, 0};
    float *d_AL, *d_cost;
    int *d_Y;
    cudaMalloc(&d_AL, n_samples*sizeof(float));
    cudaMalloc(&d_Y, n_samples*sizeof(int));
    cudaMalloc(&d_cost, sizeof(float));
    cudaMemcpy(d_AL, AL, n_samples*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, n_samples*sizeof(int), cudaMemcpyHostToDevice);
    ComputeCostCuda(d_AL, d_Y, n_samples, cost, d_cost);
    float expectedCost = 0.40666795761f;
    std::cout << "cost: " << *cost << '\n';
    float epsilon = 1.0e-7;
    if (!isless(abs(*cost-expectedCost), epsilon))
        throw "TestComputeCostCuda failed";
}

void TestBackwardCuda() {
    int *layer_dims = new int[3] {3, 2, 1};
    int num_layers {2};
    int n_samples {1};

    float *X = new float[3] {0.7f, 0.45f, 0.9f};
    int *Y = new int {1};
    float *W = new float[8] {1.2f, -1.8f, -0.7f, 0.14f, 0.3f, 0.85f, -0.56f, 0.34f};
    float *B = new float[3] {0.1f, 0.2f, 0.67f};
    float *d_X, *d_W, *d_B;
    int *d_Y;
    cudaMalloc(&d_X, 3*sizeof(float));
    cudaMalloc(&d_Y, sizeof(int));
    cudaMalloc(&d_W, 8*sizeof(float));
    cudaMalloc(&d_B, 3*sizeof(float));
    cudaMemcpy(d_X, X, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 3*sizeof(float), cudaMemcpyHostToDevice);
    
    int *W_index = new int[num_layers+1]{};
    int *B_index = new int[num_layers+1]{};
    int *Z_index = new int[num_layers+1]{};
    VectorStartIndex(layer_dims, num_layers, n_samples, W_index, B_index, Z_index);

    float *oneVec = new float[n_samples] {};
    for (int i = 0; i < n_samples; ++i)
        oneVec[i] = 1.0f;
    float *d_oneVec;
    cudaMalloc(&d_oneVec, sizeof(float));
    cudaMemcpy(d_oneVec, oneVec, sizeof(float), cudaMemcpyHostToDevice);

    float *dW = new float[8] {};
    float *dB = new float[8] {};
    float *d_Z, *d_A, *d_dZ, *d_dA, *d_dW, *d_dB;
    cudaMalloc(&d_Z, Z_index[num_layers]*sizeof(float));
    cudaMalloc(&d_A, 3*sizeof(float));
    cudaMalloc(&d_dZ, 3*sizeof(float));
    cudaMalloc(&d_dA, 3*sizeof(float));
    cudaMalloc(&d_dW, 8*sizeof(float));
    cudaMalloc(&d_dB, 3*sizeof(float));

    ForwardCuda(d_X, d_W, d_B, d_oneVec, layer_dims, num_layers, n_samples, W_index, B_index, Z_index, d_Z, d_A);

    BackwardCuda(d_W, d_B, d_Z, d_A, d_X, d_Y, d_oneVec, layer_dims, num_layers, n_samples,
                 W_index, B_index, Z_index, d_dW, d_dB, d_dZ, d_dA);

    cudaMemcpy(dW, d_dW, W_index[num_layers]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dB, d_dB, B_index[num_layers]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_W); cudaFree(d_B); cudaFree(d_oneVec); 
    cudaFree(d_Z); cudaFree(d_A); cudaFree(d_dZ); cudaFree(d_dA); cudaFree(d_dW); cudaFree(d_dB);

    float *expected_dW = new float[8] {0.179496767558757f, 0.0f, 0.115390779144915f, 0.0f, 0.230781558289831f, 0.0f,
                                     -0.409820425931346f, 0.0f};
    float *expected_dB = new float[3] {0.256423953655368f, 0.0f, -0.457899917241728f};

    float epsilon = 1.0e-7;
    for (int i = 0; i < 8; ++i) {
        if (!isless(abs(dW[i]-expected_dW[i]), epsilon))
            throw "TestBackwardCuda failed";
        std::cout << "dW[" << i << "]: " << dW[i] << '\n';
    }
    for (int i = 0; i < 3; ++i) {
        if (!isless(abs(dB[i]-expected_dB[i]), epsilon))
            throw "TestBackwardCuda failed";
        std::cout << "dB[" << i << "]: " << dB[i] << '\n';
    }

    delete[] X; delete[] Y; delete[] W; delete[] B; delete[] W_index; delete[] B_index; delete[] Z_index;
    delete[] oneVec; delete[] dW; delete[] dB;
}

void TestUpdateParamsGPU() {
    float *W = new float[8] {1.2f, -1.8f, -0.7f, 0.14f, 0.3f, 0.85f, -0.56f, 0.34f};
    float *B = new float[3] {0.1f, 0.2f, 0.67f};
    float *dW = new float[8] {0.179496767558757f, 0.0f, 0.115390779144915f, 0.0f, 0.230781558289831f, 0.0f,
                              -0.409820425931346f, 0.0f};
    float *dB = new float[3] {0.256423953655368f, 0.0f, -0.457899917241728f};

    float *d_W, *d_B, *d_dW, *d_dB;
    cudaMalloc(&d_W, 8*sizeof(float));
    cudaMalloc(&d_B, 3*sizeof(float));
    cudaMalloc(&d_dW, 8*sizeof(float));
    cudaMalloc(&d_dB, 3*sizeof(float));
    cudaMemcpy(d_W, W, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dW, dW, 8*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dB, dB, 3*sizeof(float), cudaMemcpyHostToDevice);

    UpdateParamsCuda(0.1, 8, 3, d_dW, d_dB, d_W, d_B);
    cudaMemcpy(W, d_W, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, 3*sizeof(float), cudaMemcpyDeviceToHost);

    float *expectedW = new float[8] {1.182050323244124f, -1.8f, -0.711539077914491f, 0.14f, 0.276921844171017f, 0.85f,
                                     -0.519017957406865f, 0.34f};
    float *expectedB = new float[3] {0.0743576046344633f, 0.2f, 0.715789991724173f};

    float epsilon {1.0e-7};
    for (int i = 0; i < 8; ++i) {
        if (!isless(abs(W[i]-expectedW[i]), epsilon))
            throw "TestUpdateParamsGPU failed";
        //std::cout << "W[" << i << "]: " << W[i] << '\n';
    }
    for (int i = 0; i < 3; ++i) {
        if (!isless(abs(B[i]-expectedB[i]), epsilon))
            throw "TestUpdateParamsGPU failed";
        //std::cout << "B[" << i << "]: " << B[i] << '\n';
    }
    
    cudaFree(d_W); cudaFree(d_B); cudaFree(d_dW); cudaFree(d_dB);
    delete[] W; delete[] B; delete[] dW; delete[] dB;
}

int main() {
    try {
        TestSigmoidCuda();
        TestReluCuda();
        TestVectorStartIndex();
        TestForwardCuda();
        TestInitParamsCuda();
        TestComputeCostCuda();
        TestBackwardCuda();
        TestUpdateParamsGPU();
    }
    catch (const char* exception) {
        std::cerr << "Error: " << exception << std::endl;
    }

    return 0;
}
