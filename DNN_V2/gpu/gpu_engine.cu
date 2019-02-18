#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include "activation_func.h"
#include "adam.h"
#include "backward.h"
#include "compute_cost.h"
#include "dropout.h"
#include "forward.h"
#include "initialize.h"
#include "momentum.h"
#include "setup.h"
#include "update.h"
#include <iostream>

namespace py = pybind11;
using std::string;

extern const int nThreads {128};

static int n_layers;
static int *layer_dims;
static int *W_index;
static int *B_index;
static int *Z_index;

static int n_samples;
static int batch_size;
static int n_loops;
static int last_batch_size;

static bool he;
static float learn_rate;
static string optimizer;
static float *cost;
static float lambda;
static float *layer_drop;
static bool dropout;

static int n_predict;
static int *Z_index_pr;

static float *d_X;
static int *d_Y;
static float *d_W, *d_B, *d_Z, *d_A;
static float *d_dW, *d_dB, *d_dZ, *d_dA;
static float *d_oneVec;
static float *d_cost_temp;
static float *d_VdW, *d_VdB, *d_SdW, *d_SdB;
static float *d_D;

void Setup(const int _n_layers,
           py::array_t<int, py::array::f_style | py::array::forcecast> _layer_dims,
           const int _n_samples, const int _batch_size, const float _learn_rate,
           const string _optimizer, const float _lambda,
           py::array_t<float, py::array::f_style | py::array::forcecast> _layer_drop) {
  n_layers = _n_layers;
  py::buffer_info buf_layer_dims = _layer_dims.request();
  layer_dims = static_cast<int*>(buf_layer_dims.ptr);

  n_samples = _n_samples;
  batch_size = _batch_size;
  if (n_samples % batch_size == 0) {
    n_loops = n_samples / batch_size;
    last_batch_size = batch_size;
  } else {
    n_loops = n_samples / batch_size + 1;
    last_batch_size = n_samples % batch_size;
  }

  std::cout << "n_samples: " << n_samples << '\n';
  std::cout << "batch_size: " << batch_size << '\n';
  std::cout << "n_loops: " << n_loops << '\n';
  std::cout << "last_batch_size: " << last_batch_size << '\n';

  W_index = new int[n_layers+1] {};
  B_index = new int[n_layers+1] {};
  Z_index = new int[n_layers+1] {};
  VectorIndex(n_layers, batch_size, layer_dims, W_index, B_index, Z_index);

  // variables with prefix d_ denote gpu device variables
  cudaMalloc(&d_X, layer_dims[0]*n_samples*sizeof(float)); // n_samples, not batch_size
  cudaMalloc(&d_Y, n_samples*sizeof(int)); // n_samples, not batch_size
  cudaMalloc(&d_W, W_index[n_layers]*sizeof(float));
  cudaMalloc(&d_dW, W_index[n_layers]*sizeof(float));
  cudaMalloc(&d_B, B_index[n_layers]*sizeof(float));
  cudaMalloc(&d_dB, B_index[n_layers]*sizeof(float));
  cudaMalloc(&d_Z, Z_index[n_layers]*sizeof(float));
  cudaMalloc(&d_dZ, Z_index[n_layers]*sizeof(float));
  cudaMalloc(&d_A, Z_index[n_layers]*sizeof(float));
  cudaMalloc(&d_dA, Z_index[n_layers]*sizeof(float));

  // initialize d_W and d_B
  he = true; // He et al. initialization
  InitializeParameters(n_layers, layer_dims, W_index, d_W, B_index, d_B, he);
  // temporary array for computing cost
  cudaMalloc(&d_cost_temp, batch_size*sizeof(float));
  // initialize one vector
  cudaMalloc(&d_oneVec, batch_size*sizeof(float));
  OneVector(batch_size, d_oneVec);

  // learning rate
  learn_rate = _learn_rate;

  // if using momentum or adam optimizer,
  // allocate additional gpu global memory
  optimizer = _optimizer;
  if (optimizer == "gd") {
    d_VdW = nullptr;
    d_VdB = nullptr;
    d_SdW = nullptr;
    d_SdB = nullptr;
  } else if (optimizer == "momentum") {
    cudaMalloc(&d_VdW, W_index[n_layers]*sizeof(float));
    cudaMalloc(&d_VdB, B_index[n_layers]*sizeof(float));
    d_SdW = nullptr;
    d_SdB = nullptr;
    // Initialize d_VdW and d_VdB
    InitializeMomentum(n_layers, d_VdW, d_VdB, W_index, B_index);
  } else if (optimizer == "adam") {
    cudaMalloc(&d_VdW, W_index[n_layers]*sizeof(float));
    cudaMalloc(&d_VdB, B_index[n_layers]*sizeof(float));
    cudaMalloc(&d_SdW, W_index[n_layers]*sizeof(float));
    cudaMalloc(&d_SdB, B_index[n_layers]*sizeof(float));
    // Initialize d_VdW, d_VdB, d_SdW and d_SdB
    InitializeAdam(n_layers, d_VdW, d_SdW, d_VdB, d_SdB, W_index, B_index);
  }

  // cost
  cost = new float {0.0f};

  // l2 regularization parameter
  lambda = _lambda;

  // if using dropout regularization,
  // allocate additional gpu global memory
  py::buffer_info buf_layer_drop = _layer_drop.request();
  layer_drop = static_cast<float*>(buf_layer_drop.ptr);
  dropout = CheckDropout(n_layers, layer_drop);
  std::cout << "dropout is : " << dropout << '\n';
  if (dropout) {
    cudaMalloc(&d_D, Z_index[n_layers] * sizeof(float));
  } else {
    d_D = nullptr;
  }
}

void Clean() {
  cudaFree(d_X);
  cudaFree(d_Y);
  //cudaFree(d_W);
  //cudaFree(d_B);
  cudaFree(d_Z);
  cudaFree(d_A);
  cudaFree(d_dW);
  cudaFree(d_dB);
  cudaFree(d_dZ);
  cudaFree(d_dA);
  cudaFree(d_oneVec);
  cudaFree(d_cost_temp);

  if (optimizer == "gb") {
    // do nothing
  } else if (optimizer == "momentum") {
    cudaFree(d_VdW);
    cudaFree(d_VdB);
  } else if (optimizer == "adam") {
    cudaFree(d_VdW);
    cudaFree(d_VdB);
    cudaFree(d_SdW);
    cudaFree(d_SdB);
  }

  if (dropout) cudaFree(d_D);

  //delete[] W_index;
  //delete[] B_index;
  //delete[] Z_index;
  //delete cost;
}

void Train(py::array_t<float, py::array::f_style | py::array::forcecast> _X,
           py::array_t<int  , py::array::f_style | py::array::forcecast> _Y,
           bool shuffle) {
  static int n_epoch {0};
  static int t {1};
  if (shuffle || n_epoch == 0) {
    py::buffer_info buf_X = _X.request();
    float *X = static_cast<float *>(buf_X.ptr);
    cudaMemcpy(d_X, X, layer_dims[0]*n_samples*sizeof(float), cudaMemcpyHostToDevice);

    py::buffer_info buf_Y = _Y.request();
    int *Y = static_cast<int *>(buf_Y.ptr);
    cudaMemcpy(d_Y, Y, n_samples*sizeof(int), cudaMemcpyHostToDevice);
  }

  int cur_batch_size {batch_size};
  int X_loc {0};
  int Y_loc {0};
  for (int i = 0; i < n_loops; ++i) {
    if (i == n_loops - 1) cur_batch_size = last_batch_size;
    X_loc = i * layer_dims[0] * batch_size;
    Y_loc = i * batch_size;

    // forward propagation
    Forward(n_layers, layer_dims, cur_batch_size, d_X+X_loc,
            d_W, d_B, d_Z, d_A, W_index, B_index, Z_index,
            d_oneVec, layer_drop, d_D);

    // compute cost
    ComputeCostBinaryClass(cur_batch_size, d_A+Z_index[n_layers-1],
                           d_Y+Y_loc, d_cost_temp, lambda,
                           W_index[n_layers], d_W, cost);

    // backward propagation
    Backward(n_layers, layer_dims, cur_batch_size,
             d_X+X_loc, d_Y+Y_loc, d_W, d_B, d_Z, d_A,
             d_dW, d_dB, d_dZ, d_dA, W_index, B_index, Z_index,
             d_oneVec, layer_drop, d_D, lambda);

    // update parameters
    if (optimizer == "gd") {
      UpdateParameters(W_index[n_layers], d_W, d_dW,
                       B_index[n_layers], d_B, d_dB, learn_rate);
    } else if (optimizer == "momentum") {
      float beta = 0.9f;
      UpdateMomentum(W_index[n_layers], d_VdW, d_W, d_dW,
                     B_index[n_layers], d_VdB, d_B, d_dB,
                     beta, learn_rate);
    } else if (optimizer == "adam") {
      float beta1 = 0.9f;
      float beta2 = 0.999f;
      UpdateAdam(W_index[n_layers], d_VdW, d_SdW, d_W, d_dW,
                 B_index[n_layers], d_VdB, d_SdB, d_B, d_dB,
                 beta1, beta2, learn_rate, t);
    }

    
    // print cost every 100 iterations
    if (n_epoch % 100 == 0 && i == n_loops-1) {
      std::cout << "Cost after epoch " << n_epoch << ": " << *cost << '\n';
    }
    ++t;
  }
  ++n_epoch;
}

py::tuple GetWeights() {
  // train output (parameters W and B) to python calling function
  auto param_W = py::array_t<float, py::array::f_style |
                             py::array::forcecast>(W_index[n_layers]);;
  py::buffer_info buf_W = param_W.request();
  float *W = static_cast<float *>(buf_W.ptr);
  cudaMemcpy(W, d_W, W_index[n_layers]*sizeof(float), cudaMemcpyDeviceToHost);

  auto param_B = py::array_t<float, py::array::f_style |
                             py::array::forcecast>(B_index[n_layers]);;
  py::buffer_info buf_B = param_B.request();
  float *B = static_cast<float *>(buf_B.ptr);
  cudaMemcpy(B, d_B, B_index[n_layers]*sizeof(float), cudaMemcpyDeviceToHost);

  return py::make_tuple(param_W, param_B);
}

py::array_t<float> Predict(const int _n_predict,
                           py::array_t<float, py::array::f_style | py::array::forcecast> _X,
                           py::array_t<int  , py::array::f_style | py::array::forcecast> _Y) {
  // input from python calling function
  n_predict = _n_predict;
  py::buffer_info buf_X = _X.request();
  float *X = static_cast<float *>(buf_X.ptr);
  cudaMalloc(&d_X, layer_dims[0]*n_predict*sizeof(float));
  cudaMemcpy(d_X, X, layer_dims[0]*n_predict*sizeof(float), cudaMemcpyHostToDevice);
  py::buffer_info buf_Y = _Y.request();
  int *Y = static_cast<int *>(buf_Y.ptr);
  cudaMalloc(&d_Y, n_predict*sizeof(int));
  cudaMemcpy(d_Y, Y, n_predict*sizeof(int), cudaMemcpyHostToDevice);

  Z_index_pr = new int[n_layers+1] {};
  VectorIndex(n_layers, n_predict, layer_dims, W_index, B_index, Z_index_pr);
  cudaMalloc(&d_Z, Z_index_pr[n_layers]*sizeof(float));
  cudaMalloc(&d_A, Z_index_pr[n_layers]*sizeof(float));
  cudaMalloc(&d_oneVec, n_predict*sizeof(float));
  OneVector(n_predict, d_oneVec);
  d_D = nullptr;

  Forward(n_layers, layer_dims, n_predict, d_X, d_W, d_B, d_Z, d_A,
          W_index, B_index, Z_index_pr, d_oneVec, layer_drop, d_D);

  auto predict_prob_numpy =
    py::array_t<float, py::array::f_style | py::array::forcecast>(n_predict);
  py::buffer_info buf_predict_prob_numpy = predict_prob_numpy.request();
  float *predict_prob = static_cast<float *>(buf_predict_prob_numpy.ptr);
  cudaMemcpy(predict_prob, d_A+Z_index_pr[n_layers-1],
             (Z_index_pr[n_layers]-Z_index_pr[n_layers-1]) * sizeof(float), cudaMemcpyDeviceToHost);

  float accuracy = ComputeAccuracyBinaryClass(n_predict, predict_prob, Y, 0.5f);
  std::cout << "Accuracy: " << accuracy << '\n' << '\n';

  cudaFree(d_X);
  cudaFree(d_Y);
  cudaFree(d_Z);
  cudaFree(d_A);
  cudaFree(d_oneVec);
  cudaFree(d_W);
  cudaFree(d_B);
  delete[] W_index;
  delete[] B_index;
  delete[] Z_index;
  delete[] Z_index_pr;
  delete cost;

  return predict_prob_numpy;
}


PYBIND11_MODULE(gpu_engine, m) {
  m.def("setup", &Setup);
  m.def("train", &Train);
  m.def("get_weights", &GetWeights);
  m.def("predict", &Predict);
  m.def("clean", &Clean); 
}
