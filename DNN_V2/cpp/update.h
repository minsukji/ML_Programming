#ifndef UPDATE_H
#define UPDATE_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using std::vector;

void UpdateParameters(vector<MatrixXf> &params,
    const vector<MatrixXf> &grads, float learn_rate);

#endif
