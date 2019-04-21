import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    upper_bound = np.sqrt(6 / (in_size + out_size))
    lower_bound = -upper_bound
    W = np.random.uniform(lower_bound, upper_bound, (in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)    

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    max_x = np.max(x, axis = 1)
    c = np.expand_dims(-max_x, axis = 1)
    res = np.exp(x + c) / np.expand_dims(np.sum(np.exp(x + c), axis = 1), axis = 1)
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = -np.sum(y * np.log(probs))
    n_examples = y.shape[0]
    pred_y = np.argmax(probs, axis = 1)
    correct = 0.0
    for i in range(n_examples):
        pred = pred_y[i]
        if y[i, pred] == 1.0:
            correct += 1.0
    acc = correct / float(n_examples)
    return loss, acc 
# probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1]])
# y = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
# loss, acc = compute_loss_and_acc(y, probs)
# print(loss)
# print(acc)

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    grad_X = np.zeros(X.shape) # n_example * in_size
    grad_W = np.zeros(W.shape) # in_size * out_size
    grad_b = np.zeros(b.shape) # out_size
    delta_ = delta * activation_deriv(post_act) # n_example * out_size
    n_examples = grad_X.shape[0]
    for idx in range(n_examples):
        dW = np.dot(np.expand_dims(X[idx, :], axis = 1), np.expand_dims(delta_[idx, :], axis = 0)) # in_size * out_size
        dX = np.dot(W, delta_[idx, :]) # in_size
        db = delta_[idx, :] # out_size
        grad_W += dW
        grad_X[idx, :] = dX
        grad_b += db

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    n_examples = x.shape[0]
    idxs = np.arange(n_examples)
    np.random.shuffle(idxs)
    for i in range(0, n_examples, batch_size):
        x_batch = x[idxs[i:i + batch_size]]
        y_batch = y[idxs[i:i + batch_size]]
        batches.append((x_batch, y_batch))
    return batches
