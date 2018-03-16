# -*- coding: utf-8 -*-
import os
import io
import datetime
import argparse
import numpy as np
import tensorflow as tf
from skimage import color
from PIL import Image

def init_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required = True)
    parser.add_argument('--model', required = True)
    parser.add_argument('--dataset', required = True)
    parser.add_argument('--colorspace', default = 'YUV')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--data_limit', type = int, default = -1)
    results = parser.parse_args()
    
    now = datetime.datetime.now()
    res_dir_name = ""
    if (results.data_limit != -1):
        res_dir_name += "short" + str(results.data_limit) + "_"
    res_dir_name += results.model + "_" + results.dataset + "_" + results.colorspace + "_bs" + str(results.batch_size) + "_run-"  + now.strftime("%Y-%m-%d_%H%M")
    res_dir = os.path.join(results.logdir, res_dir_name)
    os.makedirs(res_dir) 
    return res_dir, results.model, results.dataset, results.colorspace, results.batch_size, results.data_limit

def add_noise(images):
    images_noise = []
    for image in images: 
        noise = np.random.normal(size=(96, 96, 3))
        image_noise = np.concatenate((image, noise), axis = 2)
        images_noise.append(image_noise)
    return np.array(images_noise)

def save_models(res_dir, model_gen, model_dis, model_gan, epoch_str):
    weights_dir = os.path.join(res_dir, "weigths_epoch_" + epoch_str)
    os.makedirs(weights_dir)
    model_gen.save_weights(os.path.join(weights_dir, "model_gen.h5"))
    model_dis.save_weights(os.path.join(weights_dir, "model_dis.h5"))
    model_gan.save_weights(os.path.join(weights_dir, "model_gan.h5"))
   
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
                tf.Summary.Value(tag="batch disc acc", simple_value=gan_res[4]),
                tf.Summary.Value(tag="batch gen eacc", simple_value=gan_res[7]),
                tf.Summary.Value(tag="batch gen acc", simple_value=gan_res[8]),
                tf.Summary.Value(tag="batch gen mse", simple_value=gan_res[9]),
                tf.Summary.Value(tag="batch gen mae", simple_value=gan_res[10]),])
    return summary

def create_image_summary(image, mean, image_no):
    image[:, :, 0] += mean[0]
    image[:, :, 1] += mean[1]
    image[:, :, 2] += mean[2]
    image *= 255
    image_rgb_conv = np.clip(np.abs(color.yuv2rgb(image)), 0, 255).astype(np.uint8)
    image_bytes = Image.fromarray(image_rgb_conv, 'RGB')
    image_byte_array = io.BytesIO()
    image_bytes.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()
    image_summary = tf.Summary.Image(encoded_image_string = image_byte_array, height = image.shape[0], width = image.shape[1])
    return tf.Summary.Value(tag='%s/%d' % ("image", image_no), image = image_summary)
