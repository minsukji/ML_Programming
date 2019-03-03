#ifndef ADAM_H
#define ADAM_H

#include <Eigen/Core>
#include <vector>
#include <tuple>

using Eigen::MatrixXf;
using std::vector;
using std::tuple;

tuple<vector<MatrixXf>, vector<MatrixXf>> InitializeAdam(
    const vector<MatrixXf> &grads);

void UpdateAdam(vector<MatrixXf> &params, const vector<MatrixXf> &grads,
    vector<MatrixXf> V, vector<MatrixXf> S, const float learn_rate,
    const float beta1, const float beta2, const int t);

#endif
