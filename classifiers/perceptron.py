import numpy as np
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras import losses
import dataset

data, labels = dataset.load_cifar10_train_data()
data_test, labels_test = dataset.load_cifar10_test_data()

data = dataset.preproc_cifar(data)
data_test = dataset.preproc_cifar(data_test)

num = 10
idx = np.where(np.logical_and(labels >= 0, labels <= num - 1))
labels = labels[idx]
data = data[idx]
idx_test = np.where(np.logical_and(labels_test >= 0, labels_test <= num - 1))
labels_test = labels_test[idx_test]
data_test = data_test[idx_test]   
    
data = data / 255
data, mean = dataset.normalize_images(data)
y = keras.utils.to_categorical(labels, num)
data = np.array(data)
data_test = data_test / 255
data_test, _ = dataset.normalize_images(data_test, mean)
y_test = keras.utils.to_categorical(labels_test, num)
data_test = np.array(data_test)

inputs = Input((32, 32, 3))
flat = Flatten()(inputs)
dense1 = Dense(768, activation="relu")(flat)
dense2 = Dense(384, activation="relu")(dense1)
outputs = Dense(num, activation="softmax")(dense2)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer = "sgd", loss = losses.categorical_crossentropy, metrics = ["accuracy"])
model.fit(x=data, y=y, batch_size=32, epochs=1)

print(model.evaluate(data_test, y_test, batch_size =1))