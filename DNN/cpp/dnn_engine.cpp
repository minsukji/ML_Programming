#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <iostream>
#include <cmath>
#include <omp.h>

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::MatrixXi;
using Eigen::Ref;
using std::vector;
using std::tuple;


// Apply sigmoid function element-wise to a matrix.
MatrixXd Sigmoid(const Ref<const MatrixXd> &z)
{
    return 1.0 / (1.0 + exp(-z.array()));
}

// Apply rectified-linear-unit function element-wise to a matrix.
MatrixXd Relu(const Ref<const MatrixXd> &z)
{
    return z.array().max(0.0);
}

// Compute elelemt-wise derivative of sigmoid function of a matrix.
// For a = sigmoid(z), derivative of a with respect to z is a*(1-a).
// Thus, for sigmoid function, only a is the required argument.
MatrixXd SigmoidBackward(const Ref<const MatrixXd> &a)
{
    return a.array() * (1.0 - a.array());
}

// Compute element-wise derivative of relu function of a matrix.
MatrixXd ReluBackward(const Ref<const MatrixXd> &z)
{
    return (z.array() > 0.0).cast<double>().array() * Eigen::ArrayXXd::Constant(z.rows(), z.cols(), 1.0);
}

// Randomly initialize parameters (W matrix and b vector) of each layer of DNN.
// layer_dims contains the number of activations from input to output layer.
vector<MatrixXd> init_params(const Ref<const VectorXi> &layer_dims) {
    // num_layers of DNN does not include the input layer.
    int num_layers = layer_dims.size() - 1;
    try {
        if (num_layers < 2)
            throw "There should be at least one hidden layer in DNN";
    }
    catch (const char* exception) {
        std::cerr << "Error: " << exception << '\n';
    }

    vector<MatrixXd> params;

    for (int l = 1; l <= num_layers; ++l) {
        //params.push_back(MatrixXd::Random(layer_dims(l), layer_dims(l-1))* 0.01); // W matrix
        params.push_back(MatrixXd::Random(layer_dims(l), layer_dims(l-1))/std::sqrt(static_cast<double>(layer_dims(l-1)))); // W matrix
        params.push_back(MatrixXd::Zero(layer_dims(l), 1)); // b vector
    }

    return params;
}

// Carry out forward propagation.
// Arguments are matrix X (input data) and params.
tuple<vector<MatrixXd>, vector<MatrixXd>> forward(const Ref<const MatrixXd> &X, const vector<MatrixXd> &params)
{
    int num_layers {static_cast<int>(params.size()) / 2};
    vector<MatrixXd> Z, A;

    int l {0};
    Z.push_back((params[l * 2] * X).colwise() + params[l * 2 + 1].col(0));
    A.push_back(Relu(Z[l]));

    for (l = 1; l < num_layers - 1; ++l)
    {
        Z.push_back((params[l * 2] * A[l - 1]).colwise() + params[l * 2 + 1].col(0));
        A.push_back(Relu(Z[l]));
    }

    l = num_layers - 1;
    Z.push_back((params[l * 2] * A[l - 1]).colwise() + params[l * 2 + 1].col(0));
    A.push_back(Sigmoid(Z[l]));

    return std::make_tuple(Z, A);
}

// Given AL (output layer activation) and Y (), compute the cost function (cross entropy).
double compute_cost(const Ref<const MatrixXd> &AL, const Ref<const MatrixXi> &Y)
{
    int m = AL.cols();
    MatrixXd cost_matrix = (Y.cast<double>() * log(AL.array()).matrix().transpose() +
                 (1.0 - Y.cast<double>().array()).matrix() * log(1.0 - AL.array()).matrix().transpose() ) / -m;
    double cost = cost_matrix(0,0);
    return cost;
}

// Carry out backward propagation.
// Returns gradients of cost function with respect to W matrix and b vector
// In order to utilize the push_back method of vector data structure, grads
vector<MatrixXd> backward(const vector<MatrixXd> &params, const vector<MatrixXd> &Z,
                          const vector<MatrixXd> &A, const MatrixXd &X, const MatrixXi &Y)
{
    int num_layers {static_cast<int>(params.size()) / 2};
    int m {static_cast<int>(A[0].cols())};
    vector<MatrixXd> grads;
    MatrixXd dZ, dA;

    //given dA[l], compute dZ[l], dW[l], db[l] and dZ[l-1]
    int l {num_layers};
    dA = -Y.cast<double>().array() / A[l-1].array() + (1.0 - Y.cast<double>().array()) / (1.0 - A[l-1].array());

    dZ = dA.array() * SigmoidBackward(A[l - 1]).array();
    grads.push_back(dZ.rowwise().sum() / m); 
    grads.push_back(dZ * A[l - 2].transpose() / m);
    dA = params[l * 2 - 2].transpose() * dZ;

    for (l = num_layers - 1; l > 1; --l)
    {
        dZ = dA.array() * ReluBackward(Z[l-1]).array();
        grads.push_back(dZ.rowwise().sum() / m);
        grads.push_back(dZ * A[l - 2].transpose() / m);
        dA = params[l * 2 - 2].transpose() * dZ;
    }

    l = 1;
    dZ = dA.array() * ReluBackward(Z[l-1]).array();
    grads.push_back(dZ.rowwise().sum() / m);
    grads.push_back(dZ * X.transpose() / m);
    

    return grads;
}

// Update parameters
vector<MatrixXd> update_params(const vector<MatrixXd> &params, const vector<MatrixXd> &grads, double learning_rate)
{
    int num_layers {static_cast<int>(params.size()) / 2};
    vector<MatrixXd> new_params;
    for (int l = 1; l <= num_layers; ++l)
    {
        new_params.push_back(params[2*l-2] - learning_rate * grads[num_layers*2-1 - (2*l-2)]);
        new_params.push_back(params[2*l-1] - learning_rate * grads[num_layers*2-1 - (2*l-1)]);
    }

    return new_params;
}
