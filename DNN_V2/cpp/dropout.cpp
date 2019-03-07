#include <Eigen/Core>
#include <vector>
#include <random>
#include <omp.h>
#include "dropout.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;

bool CheckDropout(const Ref<const VectorXf> &layer_drop) {
  int n {static_cast<int>(layer_drop.size())};
  for (int l = 0; l < n; ++l) {
    if (layer_drop[l] < 1.0f) {
      return true;
    }
  }
  return false;
}

vector<MatrixXf> RandomlySelectDropout(
    const Ref<const VectorXi> &layer_dims, const int batch_size,
    const unsigned int seed) {
  int n {static_cast<int>(layer_dims.size())};
  vector<MatrixXf> D;

  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  for (int l = 1; l < n; ++l) { 
    D.push_back(MatrixXf::NullaryExpr(layer_dims[l], batch_size, [&](){return uniform(mt);}));
  }

  return D;
}

void ApplyDropout(const vector<MatrixXf> &D, vector<MatrixXf> &A,
    const float keep_prob) {
  int n_layers {static_cast<int>(A.size())};

  for (int l = 0; l < n_layers; ++l) {
    A[l] = (D[l].array() <= keep_prob).cast<float>() * A[l].array() / keep_prob;
  }
}
