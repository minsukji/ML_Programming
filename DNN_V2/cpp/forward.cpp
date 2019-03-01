#include <Eigen/Core>
#include <vector>
#include <tuple>
#include <omp.h>
#include "activation_func.h"
#include "forward.h"

using Eigen::MatrixXf;
using Eigen::Ref;
using std::vector;
using std::tuple;

tuple<vector<MatrixXf>, vector<MatrixXf>> Forward(
    const int n_layers, const Ref<const MatrixXf> &X,
    const vector<MatrixXf> &params, const Ref<const MatrixXf> &layer_drop) {
  vector<MatrixXf> Z, A;

  int l {0};
  Z.push_back((params[l*2] * X).colwise() + params[l*2+1].col(0));
  A.push_back(Relu(Z[l]));

  for (l = 1; l < n_layers - 1; ++l) {
    Z.push_back((params[l*2] * A[l-1]).colwise() + params[l*2+1].col(0));
    A.push_back(Relu(Z[l]));
  }

  l = n_layers - 1;
  Z.push_back((params[l*2] * A[l-1]).colwise() + params[l*2+1].col(0));
  A.push_back(Sigmoid(Z[l]));

  return std::make_tuple(Z, A);
}
