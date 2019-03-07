#ifndef FORWARD_H
#define FORWARD_H

#include <Eigen/Core>
#include <vector>
#include <tuple>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Ref;
using std::vector;
using std::tuple;

tuple<vector<MatrixXf>, vector<MatrixXf>> Forward(
    const Ref<const MatrixXf> &X, const vector<MatrixXf> &params,
    bool dropout, const Ref<const VectorXf> &layer_drop,
    vector<MatrixXf> D);

#endif
