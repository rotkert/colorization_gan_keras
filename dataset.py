import os
import glob
import pickle
import numpy as np
import paths
from scipy.misc import imread
from skimage import color

CIFAR10_PATH = paths.CIFAR10_PATH
CIFAR100_PATH = paths.CIFAR100_PATH
STL10_PATH = paths.STL10_PATH

def load_train_data(dataset, data_limit, colorspace):
    if dataset == "cifar10":
        data, _ = load_cifar10_train_data()
        data = preproc_cifar(data)
    elif dataset == "cifar100":
        data, _ = load_cifar100_train_data()
        data = preproc_cifar(data)
    elif dataset == "stl10":
        data, _ = load_stl10_train_data()
    
    data = limit_data(data, data_limit)
    data = convert_colorspace(data, colorspace)
    data, mean = normalize_images(data)
    
    return data, mean

def load_valid_data(dataset, colorspace, mean, size):
    if dataset == "cifar10":
        data, labels = load_cifar10_train_data()
        data = preproc_cifar(data)
    elif dataset == "cifar100":
        data, labels = load_cifar100_train_data()
        data = preproc_cifar(data)
    elif dataset == "stl10":
        data, labels = load_stl10_train_data()
        
    data = data[data.shape[0] - size : data.shape[0]]
    labels = labels[labels.shape[0] - size : labels.shape[0]]
    data = convert_colorspace(data, colorspace)
    data, _ = normalize_images(data, mean)
    return data, labels
    
def load_test_data(dataset, colorspace, mean):
    if dataset == "cifar10":
        data, labels = load_cifar10_test_data()
        data = preproc_cifar(data)
    elif dataset == "cifar100":
        data, labels = load_cifar100_test_data()
        data = preproc_cifar(data)
    elif dataset == "stl10":
        data, labels = load_stl10_test_data()
    
    data = convert_colorspace(data, colorspace)
    data, _ = normalize_images(data, mean)
    return data, labels

def load_cifar10_train_data():
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
    return data, labels

def load_cifar10_test_data():
    filename = '{}/test_batch'.format(CIFAR10_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'labels']
    return data_test, np.array(labels_test)

def load_cifar100_train_data():
    data, labels = [], []
    filename = '{}/train'.format(CIFAR100_PATH)
    batch_data = unpickle(filename)
    data = batch_data[b'data']
    labels = batch_data[b'fine_labels']
    return data, np.array(labels)

def load_cifar100_test_data():
    filename = '{}/test'.format(CIFAR100_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'fine_labels']
    return data_test, np.array(labels_test)

def load_stl10_train_data():
    filename = '{}/train_X.bin'.format(STL10_PATH)
    filenme_labels = '{}/train_y.bin'.format(STL10_PATH)
    with open(filename, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
    with open(filenme_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels - 1
    return images, labels

def load_stl10_test_data():
    filename = '{}/test_X.bin'.format(STL10_PATH)
    filenme_labels = '{}/test_y.bin'.format(STL10_PATH)
    with open(filename, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
    with open(filenme_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels - 1
    return images, labels
    
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preproc_cifar(data):
    data_size = data.shape[0]
    img_size = int(data.shape[1] / 3)
    data = np.dstack((data[:, :img_size], data[:, img_size:2 * img_size], data[:, 2 * img_size:]))
    return data.reshape((data_size, int(np.sqrt(img_size)), int(np.sqrt(img_size)), 3))

def limit_data(data, data_limit):
    if data_limit != -1:
        data = data[:data_limit]
    return data
        
def convert_colorspace(data, colorspace):
    if colorspace == 'YUV':
        data_yuv = color.rgb2yuv(data)
        return data_yuv
    
    elif colorspace == 'LAB':
        data_lab = color.rgb2lab(data)
        data_lab = data_lab / 100
        return data_lab
    
def normalize_images(data, mean = None):
    if mean is None:
        mean = np.mean(data, axis=tuple(range(data.ndim-1)))
   
    data[:, :, :, 0] -= mean[0]
    data[:, :, :, 1] -= mean[1]
    data[:, :, :, 2] -= mean[2]

    return data, mean
