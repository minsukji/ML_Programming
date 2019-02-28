#ifndef COMPUTE_COST_H
#define COMPUTE_COST_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::Ref;
using std::vector;

float CostL2Regularization(const int n_layers, const int batch_size,
  const vector<MatrixXf> &params, const float lambda);

float ComputeCostBinaryClass(const int n_layers, const int batch_size,
  const Ref<const MatrixXf> &AL, const Ref<const MatrixXi> &Y,
  const float lambda, const vector<MatrixXf> &params);

#endif
