#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <omp.h>
#include "initialize.h"

using Eigen::MatrixXf;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;

vector<MatrixXf> InitializeParameters(const int n_layers,
                   const Ref<const VectorXi> &layer_dims,
                   const bool he) {
  vector<MatrixXf> params;

  for (int l = 1; l <= n_layers; ++l) {
    if (he) {
      params.push_back(MatrixXf::Random(layer_dims(l), layer_dims(l-1)) *
                       std::sqrt(2.0f / static_cast<float>(layer_dims(l-1))));
    }
    else {
      params.push_back(MatrixXf::Random(layer_dims(l), layer_dims(l-1)));
    }
    params.push_back(MatrixXf::Zero(layer_dims(l), 1));
  }

  return params;
}
