import numpy as np


def calculate(x,w,b):
    temp = np.sum(x*w)+b
    if temp>0:
        return 1
    else:
        return 0

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    return calculate(x,w,b)
    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    return calculate(x,w,b)

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.3
    return calculate(x,w,b)

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

if __name__ == '__main__':
    for xs in [(0,0),(0,1),(1,0),(1,1)]:
        print(xs[0],xs[1],AND(xs[0],xs[1]),OR(xs[0],xs[1]),NAND(xs[0],xs[1]),XOR(xs[0],xs[1]))