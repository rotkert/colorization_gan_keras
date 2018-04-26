from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import tensorflow as tf
import dataset


img_width, img_height = 96, 96
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

nb_train_samples = 5000
nb_validation_samples = 8000
nb_epoch = 50
nb_classes = 10

X_train, y_train = dataset.load_stl10_train_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)

last = base_model.output
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(10, activation='sigmoid')(x)

model = Model(base_model.input, pred)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

model.summary()

model.fit(X_train, Y_train, batch_size = 32, epochs = 10)
model.save("model_nowy27042018.h5")
