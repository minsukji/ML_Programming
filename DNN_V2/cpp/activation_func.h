#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

#include <Eigen/Core>

using Eigen::MatrixXf;
using Eigen::Ref;

MatrixXf Sigmoid(const Ref<const MatrixXf> &Z);

MatrixXf Relu(const Ref<const MatrixXf> &Z);

MatrixXf SigmoidBackward(const Ref<const MatrixXf> &dA,
                         const Ref<const MatrixXf> &A);

MatrixXf ReluBackward(const Ref<const MatrixXf> &dA,
                      const Ref<const MatrixXf> &A);

#endif
