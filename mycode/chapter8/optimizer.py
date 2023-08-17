import numpy as np

class SGD:

    def __init__(self,lr = 0.01) -> None:
        self.lr = lr

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]


class Momentum:

    def __init__(self,lr = 0.01,momentum = 0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v = None


    def update(self,params,grads):
        #v会以字典型变量的形式保存与参数结构相同的数据
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


class AdaGrad:

    def __init__(self,lr = 0.01) -> None:
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr*grads[key] / (np.sqrt(self.h[key])+1e-7)


class RMSprop:

    def __init__(self,lr = 0.01,decay_rate = 0.99) -> None:
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1-self.decay_rate)*grads[key]*grads[key]
            params[key] -= self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)

class Adam:

    def __init__(self,lr = 0.01,momentum = 0.1,decay_rate=0.999) -> None:
        self.lr = lr
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.iter = 0
        self.v = None
        self.r = None

    def update(self,params,grads):
        if self.r is None:
            self.r,self.v = {},{}
            for key,val in params.items():
                self.r[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1-self.decay_rate**self.iter) / (1.0 - self.momentum**self.iter)    
        
        for key in params.keys():

            self.v[key] = self.momentum*self.v[key] + (1-self.momentum)*grads[key]
            self.r[key] = self.decay_rate*self.r[key] + (1-self.decay_rate)*grads[key]*grads[key]

            params[key] -= lr_t * self.v[key] / (np.sqrt(self.r[key])+1e-7)
