#include <Eigen/Core>
#include <vector>
#include <random>
#include "dropout.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Ref;
using std::vector;

bool CheckDropout(const Ref<const VectorXf> layer_dropout) {
  int n {static_cast<int>(layer_dropout.size())};
  for (int l = 0; l < n; ++l) {
    if (layer_dropout[l] < 1.0f) {
      return true;
    }
  }
  return false;
}

vector<MatrixXf> RandomlySelectDropout(const vector<MatrixXf> &A,
    const unsigned int seed) {
  int n_layers {static_cast<int>(A.size())};
  vector<MatrixXf> D;

  std::mt19937 mt(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  for (int l = 0; l < n_layers; ++l) { 
    D.push_back(MatrixXf::NullaryExpr(A[l].rows(), A[l].cols(), [&](){return uniform(mt);}));
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
