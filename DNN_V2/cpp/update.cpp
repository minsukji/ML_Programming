#include <Eigen/Core>
#include <vector>
#include "update.h"

using Eigen::MatrixXf;
using std::vector;

void UpdateParameters(vector<MatrixXf> &params,
    const vector<MatrixXf> &grads, const float learn_rate) {
  int n_layers {static_cast<int>(params.size()) / 2};

  for (int l = 1; l <= n_layers; ++l) {
    params[2*l-2] -= learn_rate * grads[2*n_layers-1 - (2*l-2)];
    params[2*l-1] -= learn_rate * grads[2*n_layers-1 - (2*l-1)];
  }
}
