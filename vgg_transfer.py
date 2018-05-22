from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.engine import Model
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import np_utils
import dataset

image_width, image_height = 96, 96
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))

epochs = 50
labels_num = 10
batch_size = 32

data, labels = dataset.load_stl10_train_data()
labels = np_utils.to_categorical(labels, labels_num)

last = base_model.output
flatten = Flatten()(last)
dense = Dense(256, activation='relu')(flatten)
dropout = Dropout(0.5)(dense)
output = Dense(10, activation='sigmoid')(dropout)

model = Model(base_model.input, output)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-5, momentum=0.9), metrics=['accuracy'])
model.fit(data, labels, batch_size = batch_size, epochs = epochs)

model.save("model_vgg16_stl10.h5")
