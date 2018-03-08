# -*- coding: utf-8 -*-
import os
import time
import datetime
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import io
from skimage import color
from scipy import misc
from PIL import Image

def init_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required = True)
    parser.add_argument('--model', required = True)
    parser.add_argument('--dataset', default = 'cifar10')
    parser.add_argument('--colorspace', default = 'YUV')
    parser.add_argument('--batch_size', default = 128)
    results = parser.parse_args()
    
    now = datetime.datetime.now()
    res_dir_name = results.model + "_" + results.dataset + "_" + results.colorspace + "_bs" + str(results.batch_size) + "_run-"  + now.strftime("%Y-%m-%d_%H%M")
    res_dir = os.path.join(results.run_dir, res_dir_name)
    os.makedirs(res_dir) 
    return res_dir, results.model, results.dataset, results.colorspace, results.batch_size 

def save_weights(res_dir, model_gen, model_dis, model_gan, epoch_str):
    weights_dir = os.path.join(res_dir, "weigths_epoch_" + epoch_str)
    os.makedirs(weights_dir)
    model_gen.save_weights(os.path.join(weights_dir, "weights_gen.hdf5"))
    model_dis.save_weights(os.path.join(weights_dir, "weights_dis.hdf5"))
    model_gan.save_weights(os.path.join(weights_dir, "weights_gan.hdf5"))
   
def create_summary_epoch(gan_res):
    summary = tf.Summary(value=[
                tf.Summary.Value(tag="epoch gen total loss", simple_value=gan_res[0]),
                tf.Summary.Value(tag="epoch gen loss", simple_value=gan_res[1]),
                tf.Summary.Value(tag="epoch gen L1 loss", simple_value=gan_res[2]),
                tf.Summary.Value(tag="epoch eacc", simple_value=gan_res[7]),
                tf.Summary.Value(tag="epoch acc", simple_value=gan_res[8]),
                tf.Summary.Value(tag="epoch mse", simple_value=gan_res[9]),
                tf.Summary.Value(tag="epoch mae", simple_value=gan_res[10]),])
    return summary
 
def create_summary_batch(dis_res, gan_res):
    summary = tf.Summary(value=[
                tf.Summary.Value(tag="batch disc loss", simple_value=dis_res),
                tf.Summary.Value(tag="batch gen total loss", simple_value=gan_res[0]),
                tf.Summary.Value(tag="batch gen loss", simple_value=gan_res[1]),
                tf.Summary.Value(tag="batch gen L1 loss", simple_value=gan_res[2]),
                tf.Summary.Value(tag="batch eacc", simple_value=gan_res[7]),
                tf.Summary.Value(tag="batch acc", simple_value=gan_res[8]),
                tf.Summary.Value(tag="batch mse", simple_value=gan_res[9]),
                tf.Summary.Value(tag="batch mae", simple_value=gan_res[10]),])
    return summary

def create_image_summary(image, image_no):
    image_rgb_conv = np.clip(np.abs(color.yuv2rgb(image)), 0, 255).astype(np.uint8)
    image_bytes = Image.fromarray(image_rgb_conv, 'RGB')
    image_byte_array = io.BytesIO()
    image_bytes.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()
    image_summary = tf.Summary.Image(encoded_image_string = image_byte_array, height = image.shape[0], width = image.shape[1])
    return tf.Summary.Value(tag='%s/%d' % ("image", image_no), image = image_summary)

def preproc(data, normalize=False, flip=False, mean_image=None, outType='YUV'):
    data_size = data.shape[0]
    img_size = int(data.shape[1] / 3)

    if normalize:
        if mean_image is None:
            mean_image = np.mean(data)

        mean_image = mean_image / np.float32(255)
        data = (data - mean_image) / np.float32(255)

    data_RGB = np.dstack((data[:, :img_size], data[:, img_size:2 * img_size], data[:, 2 * img_size:]))
    data_RGB = data_RGB.reshape((data_size, int(np.sqrt(img_size)), int(np.sqrt(img_size)), 3))

    if flip:
        data_RGB = data_RGB[0:data_size, :, :, :]
        data_RGB_flip = data_RGB[:, :, :, ::-1]
        data_RGB = np.concatenate((data_RGB, data_RGB_flip), axis=0)

    if outType == 'YUV':
        data_out = color.rgb2yuv(data_RGB)
        return data_out, data_RGB  # returns YUV as 4D tensor and RGB as 4D tensor

    elif outType == 'LAB':
        data_out = color.rgb2lab(data_RGB)
        data_gray = color.rgb2gray(data_RGB)[:, :, :, None]
        return data_out, data_gray  # returns LAB and grayscale as 4D tensor



def show_yuv(yuv_original, yuv_pred):
    rgb_original = np.clip(color.yuv2rgb(yuv_original), 0, 1)
    rgb_pred = np.clip(np.abs(color.yuv2rgb(yuv_pred)), 0, 1)
    grey = color.rgb2grey(yuv_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()


def show_rgb(rgb_original, rgb_pred):
    grey = color.rgb2grey(rgb_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()


def show_lab(lab_original, lab_pred):
    lab_pred = lab_pred.astype(np.float64)
    rgb_original = np.clip(color.lab2rgb(lab_original), 0, 1)
    rgb_pred = np.clip(np.abs(color.lab2rgb(lab_pred)), 0, 1)
    grey = color.rgb2grey(lab_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()
    
def save_yuv(yuv_pred, batch_no, image_no):
    rgb_pred = np.clip(np.abs(color.yuv2rgb(yuv_pred)), 0, 1)
    misc.imsave("F:\\magisterka\\results\\image_" + str(batch_no) + "_" + str(image_no) + ".png", rgb_pred)
