import numpy as np
import h5py
import scipy
import python.dnn_engine as dnn_python
import cpp.dnn_engine_pybind as dnn_cpp
import gpu.dnn_engine_pybind as dnn_gpu

# Load data set
train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
train_x_orig = np.array(train_dataset["train_set_x"][:])
train_y = np.array(train_dataset["train_set_y"][:])
train_y = np.array([train_y])

test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
test_x_orig = np.array(test_dataset["test_set_x"][:])
test_y = np.array(test_dataset["test_set_y"][:])
test_y = np.array([test_y])

# Explore the dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# 4-layer model
option = input("Enter 1 for python, 2 for eigen, 3 for cuda: ")
if option == "1":
    layer_dims = [12288, 20, 7, 5, 1]
    parameters = dnn_python.l_layer_model(train_x, train_y, layer_dims, print_cost = True)

elif option == "2":
    layer_dims = np.array([[12288], [20], [7], [5], [1]], dtype='int32', order='F')
    train_x_c = np.array(train_x, dtype='float64', order='F')
    train_y_c = np.array(train_y, dtype='int32', order='F')
    parameters = dnn_cpp.l_layer_model(train_x_c, train_y_c, layer_dims, 0.0075, 2500, True)

    test_x_c = np.array(test_x, dtype='float64', order='F')
    test_y_c = np.array(test_y, dtype='int32', order='F')
    predict = dnn_cpp.predict(test_x_c, test_y_c, parameters)

if option=="3":
    layer_dims = np.array([[12288], [20], [7], [5], [1]], dtype='int32', order='F')
    train_x_c = np.array(train_x, dtype='float32', order='F')
    #print(train_x_c.shape)
    train_y_c = np.array(train_y, dtype='int32', order='F')
    #print(train_y_c.shape)
    #parameters = dnn_cpp.l_layer_model(train_x_c, train_y_c, layer_dims, 0.0075, 3000, True)
    W, B = dnn_gpu.l_layer_model_gpu(train_x_c, train_y_c, layer_dims, 4, 0.0075, 2500, True)
    #W, B = dnn_gpu.l_layer_model_gpu(train_x_c, train_y_c, layer_dims, 4, print_cost=True)

    #W = np.array(tW, dtype='float32', order='F')
    #B = np.array(tB, dtype='float32', order='F')

    test_x_c = np.array(test_x, dtype='float32', order='F')
    test_y_c = np.array(test_y, dtype='int32', order='F')
    predict = dnn_gpu.predict(test_x_c, test_y_c, W, B, layer_dims, 4)
    #predict = dnn_gpu.predict(train_x_c, train_y_c, W, B, layer_dims, 4)
    #print(predict.shape)
    #print(predict)
    #print(test_y_c)
