#ifndef BACKWARD_H
#define BACKWARD_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using std::vector;

vector<MatrixXf> Backward(const vector<MatrixXf> &params,
    const vector<MatrixXf> &Z, const vector<MatrixXf> &A,
    const MatrixXf &X, const MatrixXi &Y, const float lambda);

#endif
