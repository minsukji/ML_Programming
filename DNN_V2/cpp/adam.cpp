#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <omp.h>
#include "adam.h"

using Eigen::MatrixXf;
using std::vector;
using std::tuple;

tuple<vector<MatrixXf>, vector<MatrixXf>> InitializeAdam(
    const vector<MatrixXf> &grads) {
  int n {static_cast<int>(grads.size())};
  vector<MatrixXf> V, S;

  // V and S have the same structure as grads
  for (int i = 0; i < n; ++i) {
    V.push_back(MatrixXf::Zero(grads[i].rows(), grads[i].cols()));
    S.push_back(MatrixXf::Zero(grads[i].rows(), grads[i].cols()));
  }

  return std::make_tuple(V, S);
}

void UpdateAdam(vector<MatrixXf> &params, const vector<MatrixXf> &grads,
    vector<MatrixXf> V, vector<MatrixXf> S, const float learn_rate,
    const float beta1, const float beta2, const int t) {
  int n {static_cast<int>(params.size()) - 1};
  static float epsilon {1e-8};
  MatrixXf V_corrected, S_corrected;

  for (int i = 0; i <= n; ++i) {
    V[n-i] = beta1 * V[n-i] + (1.0f - beta1) * grads[n-i];
    V_corrected = V[n-i] / (1.0f - pow(beta1, static_cast<float>(t)));
    S[n-i] = beta2 * S[n-i] + (1.0f - beta2) * pow(grads[n-i].array(), 2.0).matrix();
    S_corrected = S[n-i] / (1.0f - pow(beta2, static_cast<float>(t)));
    params[i] -= (learn_rate * V_corrected.array() / (sqrt(S_corrected.array()) + epsilon)).matrix();
  }
}
