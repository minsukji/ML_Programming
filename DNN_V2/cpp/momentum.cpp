#include <Eigen/Core>
#include <vector>
#include <omp.h>
#include "momentum.h"

using Eigen::MatrixXf;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;

vector<MatrixXf> InitializeMomentum(const Ref<const VectorXi> &layer_dims) {
    int n {static_cast<int>(layer_dims.size()) - 1};
  vector<MatrixXf> V;

  for (int l = n; l > 0 ; --l) {
    V.push_back(MatrixXf::Zero(layer_dims[l], 1));
    V.push_back(MatrixXf::Zero(layer_dims[l], layer_dims[l-1]));
  }
  return V;
}

void UpdateMomentum(vector<MatrixXf> &params, const vector<MatrixXf> &grads,
    vector<MatrixXf> &V, const float learn_rate, const float beta) {
  int n {static_cast<int>(params.size()) - 1};

  for (int i = 0; i <= n; ++i) {
    V[n-i] = beta * V[n-i] + (1.0f - beta) * grads[n-i];
    params[i] -= learn_rate * V[n-i];
  }
}
