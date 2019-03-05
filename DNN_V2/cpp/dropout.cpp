#include <Eigen/Core>
#include <vector>
#include <random>
#include "dropout.h"

using Eigen::MatrixXf;
using std::vector;

vector<MatrixXf> RandomlySelectDropout(const vector<MatrixXf> & A,
    const unsigned int seed) {
  int n_layers {static_cast<int>(A.size())};
  vector<MatrixXf> D;

  std::mt19937 mt(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  for (l = 0; l < n_layers; ++l) { 
    D.push_back(MatrixXf::NullaryExpr(A[l].rows(), A[l].cols(), [](){return normal(mt);}));
  }

  return D;
}

void ApplyDropout(const vector<MatrixXf> D, vector<MatrixXf> A,
    const float keep_prob) {
  int n_layers {};

  for (int l = 0; l < n_layers; ++l) {
    A[l] = (D[l].array() <= keep_prob).cast<float>() * A[l].array() / keep_prob;
  }
}
