import os
import glob
import pickle
import numpy as np
from scipy.misc import imread
from skimage import color

CIFAR10_PATH = '..\\dataset\\cifar-10-batches-py'
CIFAR100_PATH = '..\\dataset\\cifar-100-python'
STL10_PATH = '..\\dataset\\stl10_binary'

def load_train_data(dataset, data_limit, colorspace, normalize = False):
    if dataset == "cifar10":
        data = load_cifar10_train_data()
        data = preproc_cifar(data)
    elif dataset == "cifar100":
        data = load_cifar100_train_data()
        data = preproc_cifar(data)
    elif dataset == "stl10":
        data = load_stl10_train_data()
    
    data = limit_data(data, data_limit)
    data_pred, data_cond = convert_colorspace(data, colorspace)
    data_pred, data_cond, mean_data_pred, mean_data_cond = normalize_images(data_pred, data_cond, normalize)
    return data_pred, data_cond, mean_data_pred, mean_data_cond
    
def load_test_data(dataset, data_limit, colorspace, normalize=False):
    if dataset == "cifar10":
        data = load_cifar10_test_data()
        data = preproc_cifar(data)
    elif dataset == "cifar100":
        data = load_cifar100_test_data()
        data = preproc_cifar(data)
    elif dataset == "stl10":
        data = load_stl10_test_data()
    
    data = limit_data(data, data_limit)
    data_pred, data_cond = convert_colorspace(data, colorspace)
    data_pred, data_cond, _, _ = normalize_images(data_pred, data_cond, normalize)
    return data_pred, data_cond

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
    return data

def load_cifar10_test_data():
    filename = '{}/test_batch'.format(CIFAR10_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'labels']
    return data_test

def load_cifar100_train_data():
    data, labels = [], []
    filename = '{}/train'.format(CIFAR100_PATH)
    batch_data = unpickle(filename)
    data = batch_data[b'data']
    labels = batch_data[b'fine_labels']
    return data

def load_cifar100_test_data():
    filename = '{}/test'.format(CIFAR100_PATH)
    batch_data = unpickle(filename)
    data_test = batch_data[b'data']
    labels_test = batch_data[b'fine_labels']
    return data_test

def load_stl10_train_data():
    filename = '{}/train_X.bin'.format(STL10_PATH)
    with open(filename, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images
    
def load_stl10_test_data():
    filename = '{}/test_X.bin'.format(STL10_PATH)
    with open(filename, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images
    
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
        return data_yuv, data
    
    elif colorspace == 'LAB':
        data_lab = color.rgb2lab(data_RGB)
        data_gray = color.rgb2gray(data_RGB)[:, :, :, None]
        return data_lab, data_gray
    
def normalize_images(data_pred, data_cond, normalize, mean_data_pred = None, mean_data_cond = None):
    if normalize:
        if mean_image is None:
            mean_image = np.mean(data)
    
        mean_image = mean_image / np.float32(255)
        data = (data - mean_image) / np.float32(255)
    return data_pred, data_cond, mean_data_pred, mean_data_cond
