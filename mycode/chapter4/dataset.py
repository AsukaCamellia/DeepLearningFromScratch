import numpy as np
import sys,os
import urllib.request
import pickle
import os.path
import gzip
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

def download_file(file_name):
    file_path = os.path.join(dataset_path,file_name)

    if os.path.exists(file_path):
        return
    print('Download file:',file_name)
    urllib.request.urlretrieve(url_base+file_name,file_path)
    print('Done')

def download_mnist():
    for value in key_file.values():
        download_file(value)

def load_img(file_name):
    file_path = os.path.join(dataset_path,file_name)
    print('Converting '+file_name+'to np array')
    with gzip.open(file_path,'rb') as f:
        data = np.frombuffer(f.read(),np.uint8,offset=16)

    data = data.reshape(-1,img_size)
    print('Done')

    return data

def load_label(file_name):
    file_path = os.path.join(dataset_path,file_name)
    print('Convert '+file_name+'to np array')

    with gzip.open(file_path,'rb') as f:
        label = np.frombuffer(f.read(),np.uint8,offset=8)

    print('Dnoe')
    return label

def convert_numpy():
    dataset = {}
    dataset['train_img'] = load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])

    return dataset

def init_mnist():
    download_mnist()
    dataset = convert_numpy()
    print('Create pickle file')
    with open(save_file,'wb') as f:
        pickle.dump(dataset,f,-1)
    print('Done')

def change_ont_hot(x):
    t = np.zeros((x.size,10))
    for idx,row in enumerate(t):
        row[x[idx]] = 1
    return t

def load_mnist(normalize = True,flatten = True ,one_hot = False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file,'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img','test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot:
        for key in ('train_label','test_label'):
            dataset[key] = change_ont_hot(dataset[key])

    if not flatten:
        for key in ('train_img','test_img'):
            dataset[key] = dataset[key].reshape(-1,1,28,28)

    return dataset['train_img'],dataset['train_label'],dataset['test_img'],dataset['test_label']

