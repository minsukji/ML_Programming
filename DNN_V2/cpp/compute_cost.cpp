#include <Eigen/Core>
#include <vector>
#include "compute_cost.h"

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::Ref;
using std::vector;

float CostL2Regularization(const int n_layers, const int batch_size,
  const vector<MatrixXf> &params, const float lambda) {
  float result {0.0};
  for (int l = 0; l < n_layers; ++l) {
    result += params[l*2].squaredNorm();
  }
  result *= 0.5f * lambda / static_cast<float>(batch_size);
  return result;
}

float ComputeCostBinaryClass(const int n_layers, const int batch_size,
  const Ref<const MatrixXf> &AL, const Ref<const MatrixXi> &Y,
  const float lambda, const vector<MatrixXf> &params) {
  MatrixXf cost_vector = (Y.cast<float>() * log(AL.array()).matrix().transpose() +
                          (1.0f - Y.cast<float>().array()).matrix() *
                            log(1.0f - AL.array()).matrix().transpose())/-batch_size;

  if (lambda != 0.0f)
    return cost_vector(0,0) + CostL2Regularization(n_layers, batch_size, params, lambda);
  else
    return cost_vector(0,0);
}
