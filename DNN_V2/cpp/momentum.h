#ifndef MOMENTUM_H
#define MOMENTUM_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;

vector<MatrixXf> InitializeMomentum(const Ref<const VectorXi> &layer_dims);

void UpdateMomentum(vector<MatrixXf> &params, const vector<MatrixXf> &grads,
    vector<MatrixXf> &V, const float learn_rate, const float beta);

#endif
