import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step(x):
    y = x>0
    return y.astype(int)

def relu(x):
    return np.maximum(x,0)

if __name__ == '__main__':
    x = np.arange(-5,5,0.1)
    y1 = sigmoid(x)
    y2 = step(x)
    y3 = relu(x)
    plt.plot(x,y1)
    plt.plot(x,y2,'k--')
    plt.plot(x,y3)
    plt.show()