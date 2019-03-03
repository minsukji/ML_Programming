#include <Eigen/Core>
#include <vector>
#include <omp.h>
#include "momentum.h"

using Eigen::MatrixXf;
using std::vector;

vector<MatrixXf> InitializeMomentum(const vector<MatrixXf> &grads) {
  int n {static_cast<int>(grads.size())};
  vector<MatrixXf> V;

  // V has the same structure as grads
  for (int i = 0; i < n ; ++i) {
    V.push_back(MatrixXf::Zero(grads[i].rows(), grads[i].cols()));
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
