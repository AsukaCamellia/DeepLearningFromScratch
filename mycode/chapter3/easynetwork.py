import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def init_network():
    network = {}
    network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['w2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['w3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network,x):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    z1 = sigmoid(np.dot(x,w1)+b1)
    z2 = sigmoid(np.dot(z1,w2)+b2)
    y = np.dot(z2,w3)+b3
    
    return y

if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0,0.5])
    y = forward(network,x)
    print(y)