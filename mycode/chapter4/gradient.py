import numpy as np
import matplotlib.pyplot as plt
from function import *


def numercial_gradient(f,x):
    h = 1e-4
    grads = np.zeros_like(x)

    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        temp = x[idx]
        x[idx] = temp+h
        fxh1 = f(x)
        x[idx] = temp-h
        fxh2 = f(x)

        x[idx] = temp
        grads[idx] = (fxh1-fxh2)/(2*h)
        it.iternext()

    return grads

def numercial_gradient_descent(f,init_x,lr = 0.01,step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numercial_gradient(f,x)
        x -= lr*grad
    
    return x,np.array(x_history)

def f(x):
    return x[0]**2+x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = numercial_gradient_descent(f, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
print(x_history.shape)