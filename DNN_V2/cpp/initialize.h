#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;

vector<MatrixXf> InitializeParameters(const int n_layers,
                   const Ref<const VectorXi> &layer_dims,
                   const bool he);

#endif
