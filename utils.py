# -*- coding: utf-8 -*-
import os
import io
import datetime
import argparse
import numpy as np
import tensorflow as tf
import model_simple
import model_transp
import model_no_down
import model_pool_max
import model_pool_avg
import model_pool_avg_transp
import model_transp_lsgan
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

def create_models(model, size, learning_rate, momentum, lambda1, lambda2):
    if (model == "model_simple"):
        model_gen, model_dis, model_gan = model_simple.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    elif (model == "model_transp"):
        model_gen, model_dis, model_gan = model_transp.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    elif (model == "model_no_down"):
        model_gen, model_dis, model_gan = model_no_down.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    elif (model == "model_pool_max"):
        model_gen, model_dis, model_gan = model_pool_max.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    elif (model == "model_pool_avg"):
        model_gen, model_dis, model_gan = model_pool_avg.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    elif (model == "model_transp_lsgan"):
        model_gen, model_dis, model_gan = model_transp_lsgan.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    elif (model == "model_pool_avg_transp"):
        model_gen, model_dis, model_gan = model_pool_avg_transp.create_models(
            input_shape_gen = (size, size, 4),
            input_shape_dis = (size, size, 3),
            output_channels=2,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda1, lambda2])
    return model_gen, model_dis, model_gan
    
def add_noise(images):
    images_noise = []
    for image in images: 
        noise = np.random.normal(size=(32, 32, 3))
        image_noise = np.concatenate((image, noise), axis = 2)
        images_noise.append(image_noise)
    return np.array(images_noise)

def process_after_predicted(color_pred, grey, mean, colorspace):
    image_pred = np.r_[(grey.T, color_pred.T[:1], color_pred.T[1:])].T
    image_pred[:, :, 0] += mean[0]
    image_pred[:, :, 1] += mean[1]
    image_pred[:, :, 2] += mean[2]
    if colorspace == "YUV":
        image_pred *= 255
        return np.clip(np.abs(color.yuv2rgb(image_pred)), 0, 255).astype(np.uint8)
    elif colorspace == "LAB":
        data = image_pred * 100
        data = np.clip(np.abs(color.lab2rgb(data)), 0, 1)
        data *= 255
        data = data.astype(np.uint8)
        return data

def save_weights(res_dir, model_gen, model_dis, model_gan, epoch_str):
    weights_dir = os.path.join(res_dir, "weights_epoch_" + epoch_str)
    os.makedirs(weights_dir)
    model_gen.save_weights(os.path.join(weights_dir, "weights_gen.h5"))
    model_dis.save_weights(os.path.join(weights_dir, "weights_dis.h5"))
    model_gan.save_weights(os.path.join(weights_dir, "weights_gan.h5"))
   
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
    dis_sum = (dis_res + gan_res[1]) / 2
    summary = tf.Summary(value=[
                tf.Summary.Value(tag="batch disc loss", simple_value=dis_res),
                tf.Summary.Value(tag="batch gen total loss", simple_value=gan_res[0]),
                tf.Summary.Value(tag="batch gen loss", simple_value=gan_res[1]),
                tf.Summary.Value(tag="batch gen L1 loss", simple_value=gan_res[2]),
                tf.Summary.Value(tag="batch disc loss sum", simple_value=dis_sum),
                tf.Summary.Value(tag="batch disc acc", simple_value=gan_res[4]),
                tf.Summary.Value(tag="batch gen eacc", simple_value=gan_res[7]),
                tf.Summary.Value(tag="batch gen acc", simple_value=gan_res[8]),
                tf.Summary.Value(tag="batch gen mse", simple_value=gan_res[9]),
                tf.Summary.Value(tag="batch gen mae", simple_value=gan_res[10]),])
    return summary

def create_summary_evaluation(class_acc, colorfulness):
    summary = tf.Summary(value = [
                tf.Summary.Value(tag = "epoch classification accuracy", simple_value = class_acc),
                tf.Summary.Value(tag = "epoch colorfulness", simple_value = colorfulness),])
    return summary

def create_image_summary(image, image_no, text = ""):
    image_bytes = Image.fromarray(image, 'RGB')
    image_byte_array = io.BytesIO()
    image_bytes.save(image_byte_array, format='PNG')
    image_byte_array = image_byte_array.getvalue()
    image_summary = tf.Summary.Image(encoded_image_string = image_byte_array, height = image.shape[0], width = image.shape[1])
    return tf.Summary.Value(tag='%s/%d' % ("image" + text, image_no), image = image_summary)
