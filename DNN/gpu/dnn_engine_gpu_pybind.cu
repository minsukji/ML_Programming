#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dnn_engine_gpu.h"
#include <iostream>
#include <vector>

namespace py = pybind11;

// Top most level for DNN. Call functions defined above in iteration loop
//py::array_t<float> l_layer_model_gpu(py::array_t<float, py::array::f_style | py::array::forcecast> iX,
py::tuple l_layer_model_gpu(py::array_t<float, py::array::f_style | py::array::forcecast> iX,
                            py::array_t<int, py::array::f_style | py::array::forcecast> iY,
                            py::array_t<int, py::array::f_style | py::array::forcecast> ilayer_dims,
                            int num_layers, float learning_rate=0.0075, int num_iterations=3000, bool print_cost=false)
{
    // X and Y matrices.
    // These come from Python calling function, and are copied to GPU memory.
    py::buffer_info bufX = iX.request();
    py::buffer_info bufY = iY.request();
    py::buffer_info bufLayer_dims = ilayer_dims.request();
    int n_samples = static_cast<int>(bufX.shape[1]);
    float *X = static_cast<float *>(bufX.ptr);
    int *Y = static_cast<int *>(bufY.ptr);
    int *layer_dims = static_cast<int *>(bufLayer_dims.ptr);

    // W, dW, Z, dZ, A, dA matrices & B, dB vector.
    // These are allocated on GPU memory for computation.
    // Only W matrix and B vector will be copied back to CPU memory at the end to return to Python calling function.
    // #_index[l-1] is the starting index of l layer in the # vector/matrix.
    // #_index[num_layers] is the total number of elements in the # vector/matrix.
    int *W_index = new int[num_layers+1]{};
    int *B_index = new int[num_layers+1]{};
    int *Z_index = new int[num_layers+1]{};
    VectorStartIndex(layer_dims, num_layers, n_samples, W_index, B_index, Z_index);

    float *oneVec = new float[n_samples]{};
    std::fill(oneVec, oneVec+n_samples, 1.0f);

    float *d_X, *d_oneVec, *d_W, *d_B, *d_Z, *d_A, *d_dW, *d_dB, *d_dZ, *d_dA, *d_cost;
    int *d_Y;
    cudaMalloc(&d_X,  layer_dims[0] * n_samples * sizeof(float));
    cudaMalloc(&d_Y,  n_samples * sizeof(int));
    cudaMalloc(&d_oneVec, n_samples * sizeof(float));
    cudaMalloc(&d_W,  W_index[num_layers] * sizeof(float));
    cudaMalloc(&d_B,  B_index[num_layers] * sizeof(float));
    cudaMalloc(&d_Z,  Z_index[num_layers] * sizeof(float));
    cudaMalloc(&d_A,  Z_index[num_layers] * sizeof(float));
    cudaMalloc(&d_dW, W_index[num_layers] * sizeof(float));
    cudaMalloc(&d_dB, B_index[num_layers] * sizeof(float));
    cudaMalloc(&d_dZ, Z_index[num_layers] * sizeof(float));
    cudaMalloc(&d_dA, Z_index[num_layers] * sizeof(float));
    //cudaMalloc(&d_cost, sizeof(float)); // THIS IS FOR USING MANUAL CUDA COST CALCULATION (ComputeCostCuda2)
    cudaMalloc(&d_cost, n_samples * sizeof(float)); // THIS IS FOR USING CUBLAS (ComputeCostCuda3)
    

    cudaMemcpy(d_X, X, layer_dims[0] * n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, n_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oneVec, oneVec, n_samples * sizeof(float), cudaMemcpyHostToDevice);

    float *cost = new float {0};
    std::vector<float> costs;

    InitParamsCuda(layer_dims, num_layers, W_index, d_W, d_B);
    //InitParamsSerial(layer_dims, num_layers, W_index, d_W, d_B);
    for (int i = 0; i < num_iterations; ++i)
    {
        ForwardCuda(d_X, d_W, d_B, d_oneVec, layer_dims, num_layers, n_samples, W_index, B_index, Z_index, d_Z, d_A);

        ComputeCostCuda(d_A+Z_index[num_layers-1], d_Y, n_samples, cost, d_cost);
        //ComputeCostSerial(d_A+Z_index[num_layers-1], d_Y, n_samples, cost, d_cost);

        BackwardCuda(d_W, d_B, d_Z, d_A, d_X, d_Y, d_oneVec, layer_dims, num_layers, n_samples, W_index, B_index, Z_index,
                     d_dW, d_dB, d_dZ, d_dA);

        UpdateParamsCuda(learning_rate, W_index[num_layers], B_index[num_layers], d_dW, d_dB, d_W, d_B);

        if (print_cost && i % 100 == 0) {
            std::cout << "Cost after iteration " << i << ": " << *cost << '\n';
            costs.push_back(*cost);
        }
    }

    auto param_W = py::array_t<float, py::array::f_style | py::array::forcecast>(W_index[num_layers]);
    auto param_B = py::array_t<float, py::array::f_style | py::array::forcecast>(B_index[num_layers]);
    py::buffer_info bufW = param_W.request();
    py::buffer_info bufB = param_B.request();
    float *W = static_cast<float *>(bufW.ptr);
    float *B = static_cast<float *>(bufB.ptr);
    cudaMemcpy(W, d_W, W_index[num_layers]*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, B_index[num_layers]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_oneVec); cudaFree(d_W); cudaFree(d_B); cudaFree(d_Z); cudaFree(d_A);
    cudaFree(d_dW); cudaFree(d_dB), cudaFree(d_dZ), cudaFree(d_dA); cudaFree(d_cost);

    delete[] W_index; delete[] B_index; delete[] Z_index; delete[] cost;

    return py::make_tuple(param_W, param_B);
}

py::array_t<float> predict(py::array_t<float, py::array::f_style | py::array::forcecast> iX,
             py::array_t<int, py::array::f_style | py::array::forcecast> iY,
             py::array_t<float, py::array::f_style | py::array::forcecast> iW,
             py::array_t<float, py::array::f_style | py::array::forcecast> iB,
             py::array_t<int, py::array::f_style | py::array::forcecast> ilayer_dims,
             int num_layers)
{
    // X and Y matrices.
    // These come from Python calling function, and are copied to GPU memory.
    py::buffer_info bufX = iX.request();
    py::buffer_info bufY = iY.request();
    py::buffer_info bufW = iW.request();
    py::buffer_info bufB = iB.request();
    py::buffer_info bufLayer_dims = ilayer_dims.request();
    int n_samples = static_cast<int>(bufX.shape[1]);
    float *X = static_cast<float *>(bufX.ptr);
    int *Y = static_cast<int *>(bufY.ptr);
    float *W = static_cast<float *>(bufW.ptr);
    float *B = static_cast<float *>(bufB.ptr);
    int *layer_dims = static_cast<int *>(bufLayer_dims.ptr);

    // W, Z, A matrices & B vector.
    // These are allocated on GPU memory for computation.
    // Only W matrix and B vector will be copied back to CPU memory at the end to return to Python calling function.
    // #_index[l-1] is the starting index of l layer in the # vector/matrix.
    // #_index[num_layers] is the total number of elements in the # vector/matrix.
    int *W_index = new int[num_layers+1]{};
    int *B_index = new int[num_layers+1]{};
    int *Z_index = new int[num_layers+1]{};
    VectorStartIndex(layer_dims, num_layers, n_samples, W_index, B_index, Z_index);

    float *oneVec = new float[n_samples]{};
    std::fill(oneVec, oneVec+n_samples, 1.0f);

    float *d_X, *d_oneVec, *d_W, *d_B, *d_Z, *d_A;
    cudaMalloc(&d_X,  layer_dims[0] * n_samples * sizeof(float));
    cudaMalloc(&d_oneVec, n_samples * sizeof(float));
    cudaMalloc(&d_W,  W_index[num_layers] * sizeof(float));
    cudaMalloc(&d_B,  B_index[num_layers] * sizeof(float));
    cudaMalloc(&d_Z,  Z_index[num_layers] * sizeof(float));
    cudaMalloc(&d_A,  Z_index[num_layers] * sizeof(float));

    cudaMemcpy(d_X, X, layer_dims[0] * n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, W_index[num_layers] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_index[num_layers] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_oneVec, oneVec, n_samples * sizeof(float), cudaMemcpyHostToDevice);

    ForwardCuda(d_X, d_W, d_B, d_oneVec, layer_dims, num_layers, n_samples, W_index, B_index, Z_index, d_Z, d_A);
    
    auto pred = py::array_t<float, py::array::f_style | py::array::forcecast>(Z_index[num_layers]);
    py::buffer_info bufPred = pred.request();
    float *p = static_cast<float *>(bufPred.ptr);
    cudaMemcpy(p, d_A, Z_index[num_layers]*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = Z_index[num_layers-1]; i < Z_index[num_layers]; ++i) {
        if (p[i] < 0.5f)
            p[i] = 0.0f;
        else
            p[i] = 1.0f;
    }

    int sum {0};
    for (int i = Z_index[num_layers-1]; i < Z_index[num_layers]; ++i) {
        if (static_cast<int>(p[i]) == Y[i-Z_index[num_layers-1]])
            sum += 1;
    }

    std::cout << "Accuracy: " << static_cast<double>(sum) / n_samples << '\n' << '\n';

    cudaFree(d_X); cudaFree(d_oneVec); cudaFree(d_W); cudaFree(d_B); cudaFree(d_Z); cudaFree(d_A);
    delete[] W_index; delete[] B_index; delete[] Z_index;

    return pred;
}

PYBIND11_MODULE(dnn_engine_pybind, m) {
    m.def("l_layer_model_gpu", &l_layer_model_gpu, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(),
          py::arg().noconvert(), py::arg("learning_rate")=0.0075, py::arg("num_iterations")=3000, py::arg("print_cost")=false);

    m.def("predict", &predict);
}
