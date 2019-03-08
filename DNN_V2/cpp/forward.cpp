#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <omp.h>
#include "activation_func.h"
#include "dropout.h"
#include "forward.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Ref;
using std::vector;
using std::tuple;

tuple<vector<MatrixXf>, vector<MatrixXf>> Forward(
    const Ref<const MatrixXf> &X, const vector<MatrixXf> &params,
    bool dropout, const Ref<const VectorXf> &layer_drop,
    vector<MatrixXf> D) {
  int n_layers {static_cast<int>(params.size()) / 2};
  vector<MatrixXf> Z, A;

  // Perform forward propagation
  for (int l = 0; l < n_layers; ++l) {
    // Compute Z
    if (l == 0) {
      Z.push_back((params[l*2] * X).colwise() + params[l*2+1].col(0));
    } else {
      Z.push_back((params[l*2] * A[l-1]).colwise() + params[l*2+1].col(0));
    }

    // Compute A
    if (l == n_layers - 1) {
      A.push_back(Sigmoid(Z[l]));
    } else {
      A.push_back(Relu(Z[l]));
    }

    // Modify A if dropout is applied
    if (dropout && layer_drop[l+1] < 1.0f) {
      float keep_prob = layer_drop[l+1];
      ApplyDropout(D[l], A[l], keep_prob);
    }
  }

  return std::make_tuple(Z, A);
}
