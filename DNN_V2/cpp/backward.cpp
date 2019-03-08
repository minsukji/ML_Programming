#include <Eigen/Core>
#include <vector>
#include <omp.h>
#include "activation_func.h"
#include "backward.h"
#include "dropout.h"

using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::VectorXf;
using Eigen::Ref;
using std::vector;

vector<MatrixXf> Backward(const vector<MatrixXf> &params,
    const vector<MatrixXf> &Z, const vector<MatrixXf> &A,
    const MatrixXf &X, const MatrixXi &Y, const float lambda,
    const bool dropout, const Ref<const VectorXf> &layer_drop,
    vector<MatrixXf> D) {
  int n_layers {static_cast<int>(params.size()) / 2};
  int batch_size {static_cast<int>(A[0].cols())};
  vector<MatrixXf> grads;
  MatrixXf dZ, dA;

  // Compute dA of the last layer
  int l {n_layers};
  dA = -Y.cast<float>().array() / A[l-1].array() +
       (1.0f - Y.cast<float>().array()) / (1.0f - A[l-1].array());

  // Backpropagation from the last layer L to the second layer
  for (l = n_layers; 1 < l; --l) {
    // Compute dZ
    if (l == n_layers) {
      dZ = SigmoidBackward(dA, A[l-1]);
    } else {
      dZ = ReluBackward(dA, Z[l-1]);
    }

    // Compute dB
    grads.push_back(dZ.rowwise().sum() / batch_size);

    // Compute dW
    if (lambda == 0.0f) {
      // dW without L2 regularization
      grads.push_back(dZ * A[l-2].transpose() / batch_size);
    } else {
      // dW with L2 regularization
      grads.push_back(dZ * A[l-2].transpose() / batch_size +
                      lambda * params[l*2-2] / batch_size);
    }

    // Compute dA
    dA = params[l*2-2].transpose() * dZ;

    // Modify dA if dropout is applied
    if (dropout && layer_drop[l-1] < 1.0f) {
      float keep_prob = layer_drop[l-1];
      ApplyDropout(D[l-2], dA, keep_prob);
    }
  }

  // Backpropagation for the first layer
  l = 1;

  // Compute dZ
  dZ = ReluBackward(dA, Z[l-1]);

  // Compute dB
  grads.push_back(dZ.rowwise().sum() / batch_size);

  // Compute dW. Use X instead of A
  if (lambda == 0.0f) {
    // dW without L2 regularization
    grads.push_back(dZ * X.transpose() / batch_size);
  } else {
    // dW with L2 regularization
    grads.push_back(dZ * X.transpose() / batch_size +
                    lambda * params[l*2-2] / batch_size);
  }

  return grads;
}
