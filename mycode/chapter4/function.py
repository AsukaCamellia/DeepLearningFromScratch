import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        max_x = np.max(x,axis=0)
        x = x - max_x
        y = np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-4
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)


    if t.size == y.size:
        t = t.argmax(axis = 1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size),t]+delta))/batch_size

def softmax_loss(x,t):
    y = softmax(x)
    return cross_entropy_error(y,t)

