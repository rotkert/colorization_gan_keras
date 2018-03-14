import numpy as np
import keras.backend as K
from keras import losses
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import concatenate
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import Conv2DTranspose


def eacc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def l1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def create_conv(filters, kernel_size, inputs, name=None, bn=True, dropout=0., strides = 1, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, strides = strides, padding=padding, kernel_initializer='he_normal', name=name)(inputs)

    if bn:
        conv = BatchNormalization()(conv)

    if activation == 'relu':
        conv = Activation(activation)(conv)
    elif activation == 'leakyrelu':
        conv = LeakyReLU()(conv)

    if dropout != 0:
        conv = Dropout(dropout)(conv)

    return conv

def create_conv_transpose(filters, kernel_size, inputs, name=None, bn=True, dropout=0., strides = 1, padding='same', activation='relu'):
    conv = Conv2DTranspose(filters, kernel_size, strides = strides, padding=padding, kernel_initializer='he_normal', name=name)(inputs)

    if bn:
        conv = BatchNormalization()(conv)

    if activation == 'relu':
        conv = Activation(activation)(conv)
    elif activation == 'leakyrelu':
        conv = LeakyReLU()(conv)

    if dropout != 0:
        conv = Dropout(dropout)(conv)

    return conv


def create_model_gen(input_shape, output_channels):
    inputs = Input(input_shape)
    conv1 = create_conv(64, (4, 4), inputs, 'conv1', strides = 2, activation='leakyrelu')
    conv2 = create_conv(128, (4, 4), conv1, 'conv2', strides = 2, activation='leakyrelu')
    conv3 = create_conv(256, (4, 4), conv2, 'conv3', strides = 2, activation='leakyrelu')
    conv4 = create_conv(512, (4, 4), conv3, 'conv4', strides = 2, activation='leakyrelu')
    conv5 = create_conv(512, (4, 4), conv4, 'conv5', strides = 2, activation='leakyrelu')
    
    conv6 = create_conv_transpose(512, (4, 4), conv5, 'conv6', strides = 2, activation='leakyrelu')
    merge6 = concatenate([conv6, conv4], axis=3)
    conv7 = create_conv_transpose(256, (4, 4), merge6, 'conv7', strides = 2, activation='leakyrelu')
    merge7 = concatenate([conv7, conv3], axis=3)
    conv8 = create_conv_transpose(128, (4, 4), merge7, 'conv8', strides = 2, activation='leakyrelu')
    merge8 = concatenate([conv8, conv2], axis=3)
    conv9 = create_conv_transpose(64, (4, 4), merge8, 'conv9', strides = 2, activation='leakyrelu')
    merge9 = concatenate([conv9, conv1], axis=3)
    conv10 = create_conv_transpose(64, (4, 4), merge9, 'conv10', strides = 2, activation='leakyrelu')
    
    conv11 = Conv2D(output_channels, (1, 1), padding='same', name='conv11', activation = 'tanh')(conv10)

    model = Model(inputs=inputs, outputs=conv11, name='generator')

    return model


def create_model_dis(input_shape):
    inputs = Input(input_shape)
    conv1 = create_conv(64, (3, 3), inputs, 'conv1', activation='leakyrelu', dropout=.8)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = create_conv(128, (3, 3), pool1, 'conv2', activation='leakyrelu', dropout=.8)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = create_conv(256, (3, 3), pool2, 'conv3', activation='leakyrelu', dropout=.8)
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = create_conv(512, (3, 3), pool3, 'conv4', activation='leakyrelu', dropout=.8)
    pool4 = MaxPool2D((2, 2))(conv4)

    conv5 = create_conv(512, (3, 3), pool4, 'conv5', activation='leakyrelu', dropout=.8)

    flat = Flatten()(conv5)
    dense6 = Dense(1, activation='sigmoid')(flat)

    model = Model(inputs=inputs, outputs=dense6, name='discriminator')

    return model

def crop():
        def func(x):
            return x[:, :, :, 0:1]
        return  Lambda(func)

def create_model_gan(input_shape, generator, discriminator):
    input = Input(input_shape)
    gen_out = generator(input)
    
    input_dis = crop()(input)
    dis_out = discriminator(concatenate([gen_out, input_dis], axis=3))

    model = Model(inputs=[input], outputs=[dis_out, gen_out], name='dcgan')

    return model


def create_models(input_shape_gen, input_shape_dis, output_channels, lr, momentum, loss_weights):
    optimizer = Adam(lr=lr, beta_1=momentum)

    model_gen = create_model_gen(input_shape=input_shape_gen, output_channels=output_channels)
    model_gen.compile(loss=losses.mean_absolute_error, optimizer=optimizer)

    model_dis = create_model_dis(input_shape=input_shape_dis)
    model_dis.trainable = False

    model_gan = create_model_gan(input_shape=input_shape_gen, generator=model_gen, discriminator=model_dis)
    model_gan.compile(
        loss=[losses.binary_crossentropy, l1],
        metrics=[eacc, 'accuracy', 'mse', 'mae'],
        loss_weights=loss_weights,
        optimizer=optimizer
    )

    model_dis.trainable = True
    model_dis.compile(
        loss=losses.binary_crossentropy,
        optimizer=optimizer)

    return model_gen, model_dis, model_gan
