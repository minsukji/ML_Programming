import numpy as np
import h5py
import scipy
#import python.dnn_engine as dnn_python
#import cpp.dnn_engine_pybind as dnn_cpp
import gpu.gpu_engine as gpu_engine

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
layer_dims = np.array([[12288], [20], [7], [5], [1]], dtype='int32', order='F')
layer_drop = np.array([[1.0], [0.8], [1.0], [1.0], [1.0]], dtype='float32', order='F')
train_x_c = np.array(train_x, dtype='float32', order='F')
print("train x data shape: ", train_x_c.shape)
train_y_c = np.array(train_y, dtype='int32', order='F')
print("train y data shape: ", train_y_c.shape)
lamda = 0.0

gpu_engine.setup(4, layer_dims, m_train, 209, 0.0075, "adam", lamda, layer_drop)

n_epochs = 2500
shuffle = False

for l in range(0, n_epochs):
  if (shuffle):
    pass
  gpu_engine.train(train_x_c, train_y_c, shuffle)

#gpu_engine.get_weights()
gpu_engine.clean()

#W, B = dnn_gpu.l_layer_model_gpu(train_x_c, train_y_c, layer_dims, layer_dropouts, 4, lamda, 0.0075, 3500, True)
test_x_c = np.array(test_x, dtype='float32', order='F')
print("test x data shape: ", test_x_c.shape)
test_y_c = np.array(test_y, dtype='int32', order='F')
print("test y data shape: ", test_y_c.shape)
predict_prob = gpu_engine.predict(test_x_c.shape[1], test_x_c, test_y_c)
#predict_prob = gpu_engine.predict(train_x_c.shape[1], train_x_c, train_y_c)
#for i in range(0,50,5):
#  print(predict_prob[i], "; ", predict_prob[i+1], "; ", predict_prob[i+2], "; ", predict_prob[i+3], "; ", predict_prob[i+4], "\n")
#layer_dropouts = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype='float32', order='F')
#predict = dnn_gpu.predict(test_x_c, test_y_c, W, B, layer_dims, layer_dropouts, 4, 0.0)
