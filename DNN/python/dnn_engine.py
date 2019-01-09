import numpy as np

def sigmoid(z):
    """
    Apply sigmoid function element-wise to a numpy array.

    Args:
        z (numpy array of number)

    Returns:
        numpy array of float
    
    """

    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    """
    Apply rectified-linear-unit function element-wise to a numpy array.

    Args:
        z (numpy array of number)

    Returns:
        numpy array of float
    """

    return np.maximum(0.0, z)

def sigmoid_backward(a):
    """
    Compute element-wise derivative of sigmoid function of a numpy array.
    For a = sigmoid(z), derivative of a with respect to z is a*(1-a). Thus,
    for sigmoid function, only a is the required argument.

    Args:
        a (numpy array of number): a is sigmoid function value,
            not input of sigmoid function.

    Returns:
        numpy array of float
    """

    return a * (1.0 - a)

def relu_backward(z):
    """
    Compute element-wise derivative of relu function of a numpy array.

    Args:
        z (numpy array of number): z is input of relu function,
            not function value.

    Returns:
        numpy array of float
    """

    return (z > 0.0).astype(float)

def init_params(layer_dims):
    """
    Randomly initialize parameters (W matrix and b vector) of each layer of DNN.

    Args:
        layer_dims (list of int): number of activations from input to output layer.

    Returns:
        params (list of float): contains W matrix and b vector from layer 1 to num_layers.
    """

    num_layers = len(layer_dims) - 1 # num_layers of DNN does not include the input layer
    params = []

    for l in range(1, num_layers+1):
        params.append(np.random.randn(layer_dims[l], layer_dims[l-1]) /np.sqrt(layer_dims[l-1]))#* 0.01) # W matrix
        params.append(np.zeros([layer_dims[l], 1])) # b vector

    return params

def forward(X, params):
    """
    Carry out forward propagation.

    Args:
        X (numpy array of ):
        params (list of numpy arrays):

    Returns:
        tuple of Z and A, where
            Z (list of numpy arrays): result of linear part of forward propagation.
            A (list of numpy arrays): activations of DNN layers.
    """

    num_layers = len(params) // 2
    Z = []
    A = []

    l = 0
    Z.append(params[l*2].dot(X) + params[l*2+1])
    A.append(relu(Z[l]))
    #print("l", "params", "A:")
    #print(l, params[l*2].shape, X.shape, params[l*2+1].shape)

    for l in range(1, num_layers-1):
        Z.append(params[l*2].dot(A[l-1]) + params[l*2+1])
        A.append(relu(Z[l]))
        #print("l", "params", "A:")
        #print(l, params[l*2].shape, A[l-1].shape, params[l*2+1].shape)

    l = num_layers - 1
    #print("l", "params", "A:")
    #print(l, params[l*2].shape, A[l-1].shape, params[l*2+1].shape)
    Z.append(params[l*2].dot(A[l-1]) + params[l*2+1])
    A.append(sigmoid(Z[l]))

    return (Z, A)


def compute_cost(AL, Y):
    """
    Compute the cost function.

    Args:
        AL ():
        Y ():

    Returns:
        cost (float)
    """

    m = AL.shape[1]
    cost = (Y.dot(np.log(AL).T) + (1.0 - Y).dot(np.log(1.0 - AL).T)) / -m

    return np.squeeze(cost)

def backward(params, Z, A, X, Y):
    """
    Carry out backward propagation of DNN.

    Args:
        params (list of numpy arrays)
        Z (list of numpy arrays)
        A (list of numpy arrays)
        X (numpy array)
        Y (numpy array)

    Returns:
        grads (list of numpy arrays)
    """

    num_layers = len(params) // 2
    m = Y.shape[1]
    grads = []

    l = num_layers
    dA = -Y / A[l-1] + (1.0 - Y) / (1.0 - A[l-1])

    dZ = dA * sigmoid_backward(A[l-1])
    grads.append(np.sum(dZ, axis=1, keepdims=True) / m)
    grads.append(dZ.dot(A[l-2].T) / m)
    dA = params[l*2-2].T.dot(dZ)

    for l in reversed(range(2, num_layers)):
        dZ = dA * relu_backward(Z[l-1])
        grads.append(np.sum(dZ, axis=1, keepdims=True) / m)
        grads.append(dZ.dot(A[l-2].T) / m)
        dA = params[l*2-2].T.dot(dZ)

    l = 1
    dZ = dA * relu_backward(Z[l-1])
    grads.append(np.sum(dZ, axis=1, keepdims=True) / m)
    grads.append(dZ.dot(X.T) / m)

    return grads

def update_params(params, grads, learning_rate):
    """
    Update parameters

    Args:
        params (list of numpy arrays):
        grads (list of numpy arrays):
        learning_rate (float):

    Returns:
        new_params(list of numpy arrays)
    """

    num_layers = len(params) // 2
    new_params = []

    for l in range(1, num_layers+1):
        new_params.append(params[2*l-2] - learning_rate * grads[num_layers*2-1 - (2*l-2)]) # W matrix
        new_params.append(params[2*l-1] - learning_rate * grads[num_layers*2-1 - (2*l-1)]) # b vector

    return new_params

def l_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Top most level for DNN. Call functions defined above in iteration loop

    Args:
        X (numpy array)
        Y (numpy array)
        layer_dims (list of int)
        num_iterations (int)
        learning_rate (float)
    Returns:

    """

    num_layers = len(layer_dims) - 1
    params = init_params(layer_dims)
    costs = []

    for i in range(0, num_iterations):
        Z, A = forward(X, params)

        cost = compute_cost(A[num_layers - 1], Y)

        grads = backward(params, Z, A, X, Y)

        params = update_params(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    return params
