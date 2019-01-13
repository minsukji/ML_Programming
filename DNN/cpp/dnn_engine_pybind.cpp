#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <omp.h>
#include "dnn_engine.h"
#include <iostream>

namespace py = pybind11;

// Top most level for DNN. Call functions defined above in iteration loop
vector<MatrixXd> l_layer_model(const Ref<const MatrixXd> &X, const Ref<const MatrixXi> &Y,
                   const Ref<const VectorXi> &layer_dims, double learning_rate=0.0075,
                   int num_iterations=3000, bool print_cost=false)
{
    int num_layers {static_cast<int>(layer_dims.size()) - 1};
    vector<MatrixXd> params = init_params(layer_dims);
    vector<double> costs;

    for (int i = 0; i < num_iterations; ++i)
    {
        vector<MatrixXd> Z, A;
        std::tie(Z, A) = forward(X, params);

        double cost = compute_cost(A[num_layers - 1], Y);

        vector<MatrixXd> grads = backward(params, Z, A, X, Y);

        params = update_params(params, grads, learning_rate);

        if (print_cost && i % 100 == 0) {
            std::cout << "Cost after iteration " << i << ": " << cost << '\n';
            costs.push_back(cost);
        }
    }

    return params;
}

// Given the input data X and parameters params, predict the output and give accuracy
MatrixXi predict(const Ref<const MatrixXd> &X, const Ref<const MatrixXi> &Y, vector<MatrixXd> &params)
{
    int m {static_cast<int>(X.cols())};

    vector<MatrixXd> Z, A;
    std::tie(Z, A) = forward(X, params);
    int n {static_cast<int>(A.size())};

    MatrixXi p = (A[n - 1].array() > 0.5).cast<int>();

    int accuracy = (p.array() == Y.array()).cast<int>().matrix().sum();
    std::cout << "Accuracy: " << static_cast<double>(accuracy) / m << '\n' << '\n';

    return p;
}

PYBIND11_MODULE(dnn_engine_pybind, m)
{
    m.def("l_layer_model", &l_layer_model, py::arg().noconvert(), py::arg().noconvert(),
          py::arg().noconvert(), py::arg("learning_rate")=0.0075,
          py::arg("num_iterations")=3000, py::arg("print_cost")=false);

    m.def("predict", &predict);
}
