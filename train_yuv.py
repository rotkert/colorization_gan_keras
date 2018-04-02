import os
import time
import numpy as np
import tensorflow as tf
import keras.backend as K
import utils
from keras.utils import generic_utils
from dataset import load_train_data, load_test_data, load_valid_data
from tensorflow.contrib.summary.summary_ops import graph
from utils_evaluation import evaluator
from utils_evaluation import calculate_colorfulness

RES_DIR, MODEL, DATASET, COLORSPACE, BATCH_SIZE, DATA_LIMIT = utils.init_train()
EPOCHS = 109
LEARNING_RATE = 0.0005
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 0

data_yuv, mean = load_train_data(dataset = DATASET, data_limit = DATA_LIMIT, colorspace = COLORSPACE)
data_valid_yuv, lables_valid = load_valid_data(dataset = DATASET, colorspace = COLORSPACE, mean = mean, size = 500)
data_test_yuv, _ = load_test_data(dataset = DATASET, colorspace = COLORSPACE, mean = mean)

data_y = data_yuv[:, :, :, :1]
data_uv = data_yuv[:, :, :, 1:]
 
data_test_y = data_test_yuv[:, :, :, :1]
data_test_uv = data_test_yuv[:, :, :, 1:]

model_gen, model_dis, model_gan = utils.create_models(MODEL, data_yuv.shape[1], LEARNING_RATE, MOMENTUM, LAMBDA1, LAMBDA2)
 
model_gen.summary()
model_dis.summary()
model_gan.summary()
evaluator = evaluator(dataset = DATASET)
 
writer = tf.summary.FileWriter(RES_DIR)
writer.add_graph(K.get_session().graph)
 
global_batch_counter = 1
for e in range(1, EPOCHS):
    batch_counter = 1
    toggle = True
    batch_total = data_yuv.shape[0] // BATCH_SIZE
    progbar = generic_utils.Progbar(batch_total * BATCH_SIZE)

    dis_res = 0
    data_y_noise = utils.add_noise(data_y)
    data_test_y_noise = utils.add_noise(data_test_y)
    while batch_counter < batch_total:
        uv_batch = data_uv[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]
        y_batch = data_y[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]
        y_batch_noise = data_y_noise[(batch_counter - 1) * BATCH_SIZE:batch_counter * BATCH_SIZE]
      
        batch_counter += 1
      
        toggle = not toggle
        if toggle:
            x_dis = np.concatenate((model_gen.predict(y_batch_noise), y_batch), axis=3)
            y_dis = np.zeros((BATCH_SIZE, 1))
            y_dis = np.random.uniform(low=0.0, high=0.2, size=BATCH_SIZE)
        else:
            x_dis = np.concatenate((uv_batch, y_batch), axis=3)
            y_dis = np.ones((BATCH_SIZE, 1))
            y_dis = np.random.uniform(low=0.8, high=1.1, size=BATCH_SIZE)
      
        dis_res = model_dis.train_on_batch(x_dis, y_dis)
      
        x_gen = y_batch_noise
        y_gen = np.ones((BATCH_SIZE, 1))
        y_gen = np.random.uniform(low=0.8, high=1.1, size=BATCH_SIZE)
        x_output = uv_batch
        gan_res = model_gan.train_on_batch(x_gen, [y_gen, x_output])
              
        progbar.add(BATCH_SIZE,
                    values=[("D loss", dis_res),
                            ("G total loss", gan_res[0]),
                            ("G loss", gan_res[1]),
                            ("G L1", gan_res[2]),
                            ("dis acc", gan_res[4]),
                            ("pacc", gan_res[7]),
                            ("acc", gan_res[8]),
                            ("mse", gan_res[9]),
                            ("mae", gan_res[10])])
      
        summary = utils.create_summary_batch(dis_res, gan_res)
        writer.add_summary(summary, global_batch_counter)
        global_batch_counter += 1
           
    if (DATA_LIMIT == -1):
        if e % 5 == 0:
            image_values = []
            for i in range (0, 50):
                y = data_test_y[i]
                y_noise = data_test_y_noise[i]
                uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
                rgb_pred = utils.process_after_predicted(uv_pred, y, mean, COLORSPACE)
                image_value = utils.create_image_summary(rgb_pred, i)
                image_values.append(image_value)
            summary = tf.Summary(value = image_values)
            writer.add_summary(summary, e)
             
        if e % 5 == 0:
            ev = model_gan.evaluate(data_test_y_noise, [np.ones((data_test_y_noise.shape[0], 1)), data_test_uv])
            ev = np.round(np.array(ev), 4)
            summary = utils.create_summary_epoch(ev)
            writer.add_summary(summary, e)
            utils.save_models(RES_DIR, model_gen, model_dis, model_gan, str(e))
    else:
        data_valid_y = data_valid_yuv[:, :, :, :1]
        data_valid_y_noise = utils.add_noise(data_valid_y)
        data_valid_uv = data_valid_yuv[:, :, :, 1:]
        
        if e % 10 == 0:
            ev = model_gan.evaluate(data_valid_y_noise, [np.ones((data_valid_y_noise.shape[0], 1)), data_valid_uv])
            ev = np.round(np.array(ev), 4)
            summary = utils.create_summary_epoch(ev)
            writer.add_summary(summary, e)
         
        if e % 10 == 0:
            image_values = []
            for i in range (0, 50):
                y = data_y[i]
                y_noise = data_y_noise[i]
                uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
                rgb_pred = utils.process_after_predicted(uv_pred, y, mean, COLORSPACE)
                image_value = utils.create_image_summary(rgb_pred, i)
                image_values.append(image_value)
            summary = tf.Summary(value = image_values)
            writer.add_summary(summary, e)
             
            valid_image_values = []
            for i in range (0, 50):
                y = data_valid_y[i]
                y_noise = data_valid_y_noise[i]
                uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
                rgb_pred = utils.process_after_predicted(uv_pred, y, mean, COLORSPACE)
                valid_image_value = utils.create_image_summary(rgb_pred, i, "_valid")
                valid_image_values.append(valid_image_value)
            summary = tf.Summary(value = valid_image_values)
            writer.add_summary(summary, e)
            
        if e % 10 == 0:
            rgb_pred_values = []
            for i in range (data_valid_yuv.shape[0]):
                y = data_valid_y[i]
                y_noise = data_valid_y_noise[i]
                uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
                rgb_pred = utils.process_after_predicted(uv_pred, y, mean, COLORSPACE)
                rgb_pred_values.append(rgb_pred)
            rgb_pred_values = np.array(rgb_pred_values)
            class_acc = evaluator.evaluate(rgb_pred_values, lables_valid)    
            colorfulness = calculate_colorfulness(rgb_pred_values)
            summary = utils.create_summary_evaluation(class_acc, colorfulness)
            writer.add_summary(summary, e)
            utils.save_weights(RES_DIR, model_gen, model_dis, model_gan, str(e))
            
