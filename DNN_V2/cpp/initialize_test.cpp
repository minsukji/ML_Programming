#include <Eigen/Core>
#include <vector>
#include <tuple>
#include "catch.hpp"
#include "initialize.h"
#include <iostream>

TEST_CASE("initialize parameters", "[initParam]") {
  // Input layer (3); First hidden layer (4); Second hidden layer (8); Output layer (1)
  int n_layers {3};
  Eigen::VectorXi layer_dims(4);
  layer_dims << 3, 4, 8, 1;
  bool he {true};
  vector<Eigen::MatrixXf> params = InitializeParameters(n_layers, layer_dims, he);
/*
  std::cout << "W[1]: " << params[0] << '\n';
  std::cout << "W[2]: " << params[2] << '\n';
  std::cout << "W[3]: " << params[4] << '\n';
  std::cout << "B[1]: " << params[1] << '\n';
  std::cout << "B[2]: " << params[3] << '\n';
  std::cout << "B[3]: " << params[5] << '\n';
*/
  REQUIRE(params.size() == 6);
  REQUIRE(std::make_tuple(params[0].rows(), params[0].cols()) == std::make_tuple(4, 3));
  REQUIRE(std::make_tuple(params[1].rows(), params[1].cols()) == std::make_tuple(4, 1));
  REQUIRE(std::make_tuple(params[2].rows(), params[2].cols()) == std::make_tuple(8, 4));
  REQUIRE(std::make_tuple(params[3].rows(), params[3].cols()) == std::make_tuple(8, 1));
  REQUIRE(std::make_tuple(params[4].rows(), params[4].cols()) == std::make_tuple(1, 8));
  REQUIRE(std::make_tuple(params[5].rows(), params[5].cols()) == std::make_tuple(1, 1));
}
