import numpy as np
import keras
import keras.backend as K
from keras import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras import losses
from keras.layers import Flatten
import dataset
import matplotlib.pyplot as plt
from skimage import color
from keras.models import load_model
  
data, labels = dataset.load_cifar10_train_data()
data_test, labels_test = dataset.load_cifar10_test_data()

labels_test = np.array(labels_test)

data = dataset.preproc_cifar(data)
data_test = dataset.preproc_cifar(data_test)

num = 10
idx = np.where(np.logical_and(labels >= 6, labels <= 7))
labels = labels[idx]
data = data[idx]
idx_test = np.where(np.logical_and(labels_test >= 6, labels_test <= 7))
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
outputs = Dense(num, activation="softmax")(flat)
model = Model(inputs=inputs, outputs=outputs)
     
      
      
model.compile(optimizer = "sgd", loss = losses.categorical_crossentropy, metrics = ["accuracy"])
     
model.fit(x=data, y=y, batch_size=1, epochs=1)

print(model.evaluate(data_test, y_test, batch_size =1))
     
model.save(".\\linear1e.h5")

model = load_model(".\\linear1e.h5")
 
weights = model.get_weights()[0]
 
images = []
for i in range(num):
    image = weights[:, i]
    image = image.reshape((32, 32, 3))
    images.append(image)
     
images = np.array(images)
 
min = np.amin(images, axis=tuple(range(images.ndim-1)))
print(min)
images[:, :, :, 0] -= min[0] 
images[:, :, :, 1] -= min[1]
images[:, :, :, 2] -= min[2]
 
max = np.amax(images, axis=tuple(range(images.ndim-1)))
print(max)
images[:, :, :, 0] /= max[0]
images[:, :, :, 1] /= max[1]
images[:, :, :, 2] /= max[2]
 
for i in range(num):
    plt.imshow(images[i])
    plt.show()
