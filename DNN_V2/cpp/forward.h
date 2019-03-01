#ifndef FORWARD_H
#define FORWARD_H

#include <Eigen/Core>
#include <vector>
#include <tuple>

using Eigen::MatrixXf;
using Eigen::Ref;
using std::vector;
using std::tuple;

tuple<vector<MatrixXf>, vector<MatrixXf>> Forward(
    const int n_layers, const Ref<const MatrixXf> &X,
    const vector<MatrixXf> &params, const Ref<const MatrixXf> &layer_drop);

#endif
