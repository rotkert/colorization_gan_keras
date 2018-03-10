import os
import glob
import pickle
import numpy as np
from scipy.misc import imread
from skimage import color
from utils import preproc

CIFAR10_PATH = 'F:\\magisterka\\datasets\\cifar-10-python\\cifar-10-batches-py'
CIFAR100_PATH = 'F:\\magisterka\\datasets\\cifar-100-python'
IMAGENET_PATH = '../../../datasets/ImageNet'


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_train_data(dataset, normalize=False, shuffle=False, flip=False, count=-1, out_type='YUV'):
    if dataset == "cifar10":
        return load_cifar10_train_data(out_type = out_type)
    elif dataset == "cifar100":
        return load_cifar100_train_data(out_type = out_type)
    
def load_test_data(dataset, normalize=False, count=-1, out_type='YUV'):
    if dataset == "cifar10":
        return load_cifar10_test_data(out_type = out_type)
    elif dataset == "cifar100":
        return load_cifar100_test_data(out_type = out_type)
    

def load_cifar10_train_data(normalize=False, shuffle=False, flip=False, count=-1, out_type='YUV'):
    names = unpickle('{}/batches.meta'.format(CIFAR10_PATH))[b'label_names']
    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(CIFAR10_PATH, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data[b'data']))
            labels = np.hstack((labels, batch_data[b'labels']))
        else:
            data = batch_data[b'data']
            labels = batch_data[b'labels']

    if shuffle:
        np.random.shuffle(data)

    if count != -1:
        data = data[:count]

    return preproc(data, normalize=normalize, flip=flip, outType = out_type)


def load_cifar10_test_data(normalize=False, count=-1, out_type='YUV'):
    filename = '{}/test_batch'.format(CIFAR10_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'labels']

    if count != -1:
        data_test = data_test[:count]

    return preproc(data_test, normalize=normalize, outType=out_type)

def load_cifar100_train_data(normalize=False, shuffle=False, flip=False, count=-1, out_type='YUV'):
    data, labels = [], []
    filename = '{}/train'.format(CIFAR100_PATH)
    batch_data = unpickle(filename)
    data = batch_data[b'data']
    labels = batch_data[b'fine_labels']

    if shuffle:
        np.random.shuffle(data)

    if count != -1:
        data = data[:count]

    return preproc(data, normalize=normalize, flip=flip, outType = out_type)

def load_cifar100_test_data(normalize=False, count=-1, out_type='YUV'):
    filename = '{}/test'.format(CIFAR100_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'fine_labels']

    if count != -1:
        data_test = data_test[:count]

    return preproc(data_test, normalize=normalize, outType=out_type)