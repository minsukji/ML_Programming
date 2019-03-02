#include <Eigen/Core>
#include <vector>
#include <omp.h>
#include "activation_func.h"
#include "backward.h"

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using std::vector;

vector<MatrixXf> Backward(const vector<MatrixXf> &params,
    const vector<MatrixXf> &Z, const vector<MatrixXf> &A,
    const MatrixXf &X, const MatrixXi &Y) {
  int n_layers {static_cast<int>(params.size()) / 2};
  int batch_size {static_cast<int>(A[0].cols())};
  vector<MatrixXf> grads;
  MatrixXf dZ, dA;

  int l {n_layers};
  dA = -Y.cast<float>().array() / A[l-1].array() +
       (1.0f - Y.cast<float>().array()) / (1.0f - A[l-1].array());

  for (l = n_layers; 1 < l; --l) {
    if (l == n_layers) {
      dZ = SigmoidBackward(dA, A[l-1]);
    } else {
      dZ = ReluBackward(dA, Z[l-1]);
    }

    grads.push_back(dZ.rowwise().sum() / batch_size);
    grads.push_back(dZ * A[l-2].transpose() / batch_size);
    dA = params[l*2-2].transpose() * dZ;
  }

  l = 1;
  dZ = ReluBackward(dA, Z[l-1]);
  grads.push_back(dZ.rowwise().sum() / batch_size);
  grads.push_back(dZ * X.transpose() / batch_size);

  return grads;
}
