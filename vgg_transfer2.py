from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(3,32,32),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()


#Then training with your data ! 

nb_train_samples = 5000
nb_validation_samples = 8000
nb_epoch = 2
nb_classes = 10

data, labels = dataset.load_cifar100_train_data()
data = dataset.preproc_cifar(data)
Y_train = np_utils.to_categorical(labels, nb_classes)


data_test, labels_test = dataset.load_cifar100_train_data()
data_test = dataset.preproc_cifar(data_test)
y_test = np_utils.to_categorical(labels_test, nb_classes)


model.fit(data, Y_train, batch_size = 32, epochs = 2)

print(model.evaluate(data_test, y_test, batch_size =1))
model.save("model_nowy.h5")
