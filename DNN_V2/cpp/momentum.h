#ifndef MOMENTUM_H
#define MOMENTUM_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using std::vector;

vector<MatrixXf> InitializeMomentum(const vector<MatrixXf> &grads);

void UpdateMomentum(vector<MatrixXf> &params, const vector<MatrixXf> &grads,
    vector<MatrixXf> &V, const float learn_rate, const float beta);

#endif
