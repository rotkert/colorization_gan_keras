import numpy as np
import keras
import utils_dataset
from keras.models import load_model

data, labels = utils_dataset.load_cifar10_train_data()
data_test, labels_test = utils_dataset.load_cifar10_test_data()

data = utils_dataset.preproc_cifar(data)
data_test = utils_dataset.preproc_cifar(data_test)

num = 10
idx = np.where(np.logical_and(labels >= 0, labels <= num - 1))
labels = labels[idx]
data = data[idx]
idx_test = np.where(np.logical_and(labels_test >= 0, labels_test <= num - 1))
labels_test = labels_test[idx_test]
data_test = data_test[idx_test]   
    
y = keras.utils.to_categorical(labels, num)
data = np.array(data)

mean = 120.707
std = 64.15
y_test = keras.utils.to_categorical(labels_test, num)
data_test = (data_test - mean) / (std + 1e-7)
data_test = np.array(data_test)

model = load_model(paths.VGG_CIFAR10_PATH)

print(model.evaluate(data_test, y_test, batch_size =1))