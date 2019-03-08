#ifndef DROPOUT_H
#define DROPOUT_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::Ref;
using std::vector;

bool CheckDropout(const Ref<const VectorXf> &layer_drop);

vector<MatrixXf> RandomlySelectDropout(
    const Ref<const VectorXi> &layer_dims, const int batch_size,
    const unsigned int seed);

void ApplyDropout(const Ref<const MatrixXf> &D, Ref<MatrixXf> A,
    const float keep_prob);

#endif
