import numpy as np
from dataset import *
from function import *
from gradient import *

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.1) -> None:
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        w1,w2 = self.params['w1'],self.params['w2']
        b1,b2 = self.params['b1'],self.params['b2']

        y1 = sigmoid(np.dot(x,w1)+b1)
        y2 = softmax(np.dot(y1,w2)+b2)

        return y2
    
    def loss(self,x,t):
        y = self.predict(x)

        return cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])

        return accuracy
    
    def num_gradient(self,x,t):
        fun_loss = lambda w:self.loss(x,t)

        grads = {}
        grads['w1'] = numercial_gradient(fun_loss,self.params['w1'])
        grads['b1'] = numercial_gradient(fun_loss,self.params['b1'])
        grads['w2'] = numercial_gradient(fun_loss,self.params['w2'])
        grads['b2'] = numercial_gradient(fun_loss,self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, w2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

if __name__ == '__main__':
    net = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)
    x_train,t_train,x_test,t_test = load_mnist(normalize=True,flatten=True,one_hot=True)

    train_size = x_train.shape[0]
    batch_size = 100

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    epoch = 100
    learn_rate = 0.1
    iter_per_epoch = int(train_size/batch_size)
    iter_sumnum = epoch * iter_per_epoch

    for i in range(iter_sumnum):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = net.gradient(x_batch,t_batch)

        for key in ('w1','b1','w2','b2'):
            net.params[key] -= learn_rate*grads[key]

        loss = net.loss(x_batch,t_batch)
        train_loss_list.append(loss)

        if i%iter_per_epoch == 0:
            train_acc = net.accuracy(x_train,t_train)
            test_acc = net.accuracy(x_test,t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('train_acc  test_acc',train_acc,test_acc)
        
    x = np.arange(epoch)
    plt.plot(x,train_acc_list,label = 'train_acc',linestyle = '--')
    plt.plot(x,test_acc_list,label = 'test_acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.ylim(0,1.0)
    plt.legend(loc = 'lower right')
    plt.show()
