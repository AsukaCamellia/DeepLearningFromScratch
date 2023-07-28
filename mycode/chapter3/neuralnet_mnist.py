# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
import numpy as np
import os.path
import gzip
import urllib.request

dataset_path = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_path,'mnist.pkl')

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

train_num = 60000
test_num = 10000
img_size = 784
img_dim = (1,28,28)


def downloadfile(file_name):
    file_path = os.path.join(dataset_path,file_name)

    if os.path.exists(file_path):
        return
    print('Downloading '+file_name)
    urllib.request.urlretrieve(url_base+file_name,file_path)
    print('Done')

def downloadmnist():
    for v in key_file.values():
        downloadfile(v)

def load_img(file_name):
    file_path = os.path.join(dataset_path,file_name)
    print('converting '+file_name+" to numpy array")
    with gzip.open(file_path,"rb") as f:
        data = np.frombuffer(f.read(),np.uint8,offset=16)
    
    data = data.reshape(-1,img_size)
    print('done')

    return data

def load_label(file_name):
    file_path = os.path.join(dataset_path,file_name)
    print('converting '+file_name+'to numpy array')
    with gzip.open(file_path,'rb') as f:
        labels = np.frombuffer(f.read(),np.uint8,offset=8)
    print('Done')

    return labels

def convert_numpy():
    dataset = {}
    dataset['train_img'] = load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])

    return dataset

def init_mnist():
    downloadmnist()
    dataset = convert_numpy()
    print('creating pickle file')
    with open(save_file,'wb') as f:
        pickle.dump(dataset,f,-1)
    print('done')

def change_one_hot(x):
    t = np.zeros((x.size,10))
    for idx,row in enumerate(t):
        row[x[idx]] = 1
    return t


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下,标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        for key in ('train_label', 'test_label'):
            dataset[key] = change_one_hot(dataset[key])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    
    return  dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label']



def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    maxn = np.max(x)
    exp_a = np.exp(x-maxn)
    sum = np.sum(exp_a)
    y = exp_a/sum

    return y

def get_data():
    x_train,t_train,x_test,t_test = load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open('D:\deeplearningfromScratch\mycode\chapter3\sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    z1 = sigmoid(np.dot(x,w1)+b1)
    z2 = sigmoid(np.dot(z1,w2)+b2)
    y = softmax(np.dot(z2,w3)+b3)
    return y


if __name__ == '__main__':
    x,t = get_data()
    network = init_network()
    batch_size = 100
    accuracy_cnt = 0
    for i in range(0,len(x),batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network,x_batch)
        p = np.argmax(y_batch,axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print(accuracy_cnt/len(x))