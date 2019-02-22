#include <Eigen/Core>
#include <cmath>
#include <omp.h>
#include "activation_func.h"

using Eigen::MatrixXf;
using Eigen::Ref;

// Compute Sigmoid function
MatrixXf Sigmoid(const Ref<const MatrixXf> &Z) {
  return 1.0f / (1.0f + exp(-Z.array()));
}

// Compute Rectified-Linear-Unite function
MatrixXf Relu(const Ref<const MatrixXf> &Z) {
  return Z.array().max(0.0);
}

// Compute derivative of Sigmoid function
// For use in Backprop, dA is multiplied to get dZ
MatrixXf SigmoidBackward(const Ref<const MatrixXf> &dA,
                         const Ref<const MatrixXf> &A) {
  // Compute and return dZ
  return dA.array() * A.array() * (1.0f - A.array());
}

// Compute derivative of Relu function
// For use in Backprop, dA is multipled to get dZ
MatrixXf ReluBackward(const Ref<const MatrixXf> &dA,
                      const Ref<const MatrixXf> &Z) {
  // Compute and return dZ
  return (Z.array() > 0.0f).cast<float>() * dA.array();
}
