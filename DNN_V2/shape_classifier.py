import numpy as np
import sklearn.datasets
import gpu.gpu_engine as gpu_engine

# Load data set
np.random.seed(3)
train_x, train_y = sklearn.datasets.make_moons(n_samples=300, noise=.2)
train_x = train_x.T
train_y = train_y.reshape((1, train_y.shape[0]))

layer_dims = np.array([[train_x.shape[0]], [5], [2], [1]], dtype='float32', order='F')
layer_drop = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]], dtype='float32', order='F')
train_x_c = np.array(train_x, dtype='float32', order='F')
print("train x data shape: ", train_x_c.shape)
train_y_c = np.array(train_y, dtype='int32', order='F')
print("train y data shape: ", train_y_c.shape)
lamda = 0.0

gpu_engine.setup(3, layer_dims, train_x.shape[1], 64, 0.0007, "adam", lamda, layer_drop)

n_epochs = 2800
shuffle = False

for l in range(0, n_epochs):
  if (shuffle):
    pass
  gpu_engine.train(train_x_c, train_y_c, shuffle)

gpu_engine.clean()

predict_prob = gpu_engine.predict(train_x_c.shape[1], train_x_c, train_y_c) 
