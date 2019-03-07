#ifndef DROPOUT_H
#define DROPOUT_H

#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Ref;
using std::vector;

bool CheckDropout(const Ref<const VectorXf> layer_dropout);

vector<MatrixXf> RandomlySelectDropout(const vector<MatrixXf> &A,
    const unsigned int seed);

void ApplyDropout(const vector<MatrixXf> &D, vector<MatrixXf> &A,
    const float keep_prob);

#endif
