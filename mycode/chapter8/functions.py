import numpy as np

def indentify_function(x):
    return x

def step_function(x):
    return np.array(x>0,dtype=np.int32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return(1-sigmoid(x))*sigmoid(x)

def relu(x):
    return np.maximum(0,x)

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x,axis=0)
        y = np.exp(x)/(np.sum(np.exp(x),axis=0))
        return y.T
    
    x = x - np.max(x,axis=0)
    y = np.exp(x)/(np.sum(np.exp(x),axis=0))
    return y

def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)

    if t.size == y.size:
        t = np.argmax(t,axis=1)

    batch_size = y.shape[0]

    error = -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
    return error

def softmax_loss(x,t):
    y = softmax(x)
    return cross_entropy_error(y,t)
