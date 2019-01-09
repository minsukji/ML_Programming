from dnn_engine import *
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

def test_sigmoid():
    z = np.array([[0.0, -1.23, 0.74], [-0.001, 2.6, 0.0002]])
    expectedResult = np.array([[0.5, 0.22618142573, 0.67699585624], [0.49975, 0.930861579657, 0.50005]])

    assert_almost_equal(expectedResult, sigmoid(z), decimal=8)

def test_relu():
    z = np.array([[0.0, -1.23, 0.74], [-0.001, 2.6, 0.0002]])
    expectedResult = np.array([[0.0, 0.0, 0.74], [0.0, 2.6, 0.0002]])

    assert_almost_equal(expectedResult, relu(z), decimal=8) 

def test_initParams1():
    layer_dims = [3, 4, 8, 1]

    params = init_params(layer_dims)

    assert_equal(6, len(params))
    assert_array_equal((4, 3), params[0].shape)
    assert_array_equal((4, 1), params[1].shape)
    assert_array_equal((8, 4), params[2].shape)
    assert_array_equal((8, 1), params[3].shape)
    assert_array_equal((1, 8), params[4].shape)
    assert_array_equal((1, 1), params[5].shape)

def test_initParams2():
    layer_dims = [1, 1, 1, 1]

    params = init_params(layer_dims)

    assert_equal(6, len(params))
    assert_array_equal((1, 1), params[0].shape)
    assert_array_equal((1, 1), params[1].shape)
    assert_array_equal((1, 1), params[2].shape)
    assert_array_equal((1, 1), params[3].shape)
    assert_array_equal((1, 1), params[4].shape)
    assert_array_equal((1, 1), params[5].shape)

def test_initParams3():
    layer_dims = [1, 1, 10]

    params = init_params(layer_dims)

    assert_equal(4, len(params))
    assert_array_equal((1, 1), params[0].shape)
    assert_array_equal((1, 1), params[1].shape)
    assert_array_equal((10, 1), params[2].shape)
    assert_array_equal((10, 1), params[3].shape)

def test_forward():
    # Input layer (3); First hidden layer (2); Output layer (1)
    # W1(2,3), B1(2,1), W2(1,2), B2(1,1)
    params = []
    X  = np.array([[0.7], [0.45], [0.9]])
    W1 = np.array([[1.2, -0.7, 0.3], [-1.8, 0.14, 0.85]])
    params.append(W1)
    B1 = np.array([[0.1], [0.2]])
    params.append(B1)
    W2 = np.array([[-0.56, 0.34]])
    params.append(W2)
    B2 = np.array([[0.67]])
    params.append(B2)

    Z, A = forward(X, params)
    Z1 = np.array([[0.895], [-0.232]])
    A1 = np.array([[0.895], [0.0]])
    Z2 = np.array([[0.16880]])
    A2 = np.array([[0.542100082758272]])

    assert_almost_equal(Z1, Z[0], decimal=8)
    assert_almost_equal(Z2, Z[1], decimal=8)
    assert_almost_equal(A1, A[0], decimal=8)
    assert_almost_equal(A2, A[1], decimal=8)

def test_computeCost():
    AL = np.array([[0.1, 0.97, 0.51, 0.3, 0.02]])
    Y  = np.array([[0, 1, 1, 1, 0]])
    expectedCost = 0.40666795761
    assert_almost_equal(expectedCost, compute_cost(AL, Y), decimal=8)

def test_backward():
    params = []
    X  = np.array([[0.7], [0.45], [0.9]])
    W1 = np.array([[1.2, -0.7, 0.3], [-1.8, 0.14, 0.85]])
    params.append(W1)
    B1 = np.array([[0.1], [0.2]])
    params.append(B1)
    W2 = np.array([[-0.56, 0.34]])
    params.append(W2)
    B2 = np.array([[0.67]])
    params.append(B2)
    Y = np.array([[1]])
    Z, A = forward(X, params)

    grads = backward(params, Z, A, X, Y)

    dB2 = np.array([[-0.457899917241728]])
    dW2 = np.array([[-0.409820425931346, 0.0]])
    dB1 = np.array([[0.256423953655368], [0.0]])
    dW1 = np.array([[0.179496767558757, 0.115390779144915, 0.230781558289831], [0.0, 0.0, 0.0]])

    assert_almost_equal(dB2, grads[0], decimal=8)
    assert_almost_equal(dW2, grads[1], decimal=8)
    assert_almost_equal(dB1, grads[2], decimal=8)
    assert_almost_equal(dW1, grads[3], decimal=8)

def test_updateParams():
    params = []
    W1 = np.array([[1.2, -0.7, 0.3], [-1.8, 0.14, 0.85]])
    params.append(W1)
    B1 = np.array([[0.1], [0.2]])
    params.append(B1)
    W2 = np.array([[-0.56, 0.34]])
    params.append(W2)
    B2 = np.array([[0.67]])
    params.append(B2)

    grads = [] 
    dB2 = np.array([[-0.457899917241728]])
    grads.append(dB2)
    dW2 = np.array([[-0.409820425931346, 0.0]])
    grads.append(dW2)
    dB1 = np.array([[0.256423953655368], [0.0]])
    grads.append(dB1)
    dW1 = np.array([[0.179496767558757, 0.115390779144915, 0.230781558289831], [0.0, 0.0, 0.0]])
    grads.append(dW1)

    new_params = update_params(params, grads, 0.1);

    new_W1 = np.array([[1.182050323244124, -0.711539077914491, 0.276921844171017], [-1.8, 0.14, 0.85]])
    new_B1 = np.array([[0.0743576046344633], [0.2]])
    new_W2 = np.array([[-0.519017957406865, 0.34]])
    new_B2 = np.array([[0.715789991724173]])

    assert_almost_equal(new_W1, new_params[0], decimal=8)
    assert_almost_equal(new_B1, new_params[1], decimal=8)
    assert_almost_equal(new_W2, new_params[2], decimal=8)
    assert_almost_equal(new_B2, new_params[3], decimal=8)
