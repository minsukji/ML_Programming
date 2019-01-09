#ifndef DNN_ENGINE_H
#define DNN_ENGINE_H

#include <Eigen/Core>
#include <vector>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;
using std::tuple;

MatrixXd Sigmoid(const Ref<const MatrixXd> &z);

MatrixXd Relu(const Ref<const MatrixXd> &z);

MatrixXd SigmoidBackward(const Ref<const MatrixXd> &a);

MatrixXd ReluBackward(const Ref<const MatrixXd> &z);

vector<MatrixXd> init_params(const Ref<const VectorXi> &layer_dims);

tuple<vector<MatrixXd>, vector<MatrixXd>>
    forward(const Ref<const MatrixXd> &X, const vector<MatrixXd> &params);

double compute_cost(const Ref<const MatrixXd> &AL, const Ref<const MatrixXi> &Y);

vector<MatrixXd> backward(const vector<MatrixXd> &params, const vector<MatrixXd> &Z,
                          const vector<MatrixXd> &A, const MatrixXd &X, const MatrixXi &Y);

vector<MatrixXd> update_params(const vector<MatrixXd> &params, const vector<MatrixXd> &grads,
                               double learning_rate);

vector<MatrixXd> l_layer_model(const Ref<const MatrixXd> &X, const Ref<const MatrixXi> &Y,
                               const Ref<const VectorXi> &layer_dims, double learning_rate,
                               int num_iterations, bool print_cost);

#endif
