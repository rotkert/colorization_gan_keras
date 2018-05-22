import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.utils import generic_utils
from tensorflow.contrib.summary.summary_ops import graph
import utils
import utils_dataset
from utils_evaluation import evaluator
from utils_evaluation import calculate_colorfulness

results_dir, model, dataset, colorspace, batch_size, data_limit = utils.init_train()
epochs = 100
learning_rate = 0.0002
momentum = 0.5
lambda1 = 1
lambda2 = 100

data, mean = utils_dataset.load_train_data(dataset = dataset, data_limit = data_limit, colorspace = colorspace)
data_valid, lables_valid = utils_dataset.load_valid_data(dataset = dataset, colorspace = colorspace, mean = mean, size = 500)
data_test, lables_test = utils_dataset.load_test_data(dataset = dataset, colorspace = colorspace, mean = mean)

data_grey = data[:, :, :, :1]
data_color = data[:, :, :, 1:]
 
data_grey_test = data_test[:, :, :, :1]
data_color_test = data_test[:, :, :, 1:]

model_gen, model_dis, model_gan = utils.create_models(model, data.shape[1], learning_rate, momentum, lambda1, lambda2)
 
model_gen.summary()
model_dis.summary()
model_gan.summary()
evaluator = evaluator(dataset = dataset)
 
writer = tf.summary.FileWriter(results_dir)
writer.add_graph(K.get_session().graph)
 
global_batch_counter = 1
for e in range(1, epochs):
    batch_counter = 1
    toggle = True
    batch_total = data.shape[0] // batch_size
    progbar = generic_utils.Progbar(batch_total * batch_size)

    dis_results = 0
    data_grey_noise = utils.add_noise(data_grey)
    while batch_counter < batch_total:
        batch_color = data_color[(batch_counter - 1) * batch_size:batch_counter * batch_size]
        batch_grey = data_grey[(batch_counter - 1) * batch_size:batch_counter * batch_size]
        batch_grey_noise = data_grey_noise[(batch_counter - 1) * batch_size:batch_counter * batch_size]
      
        batch_counter += 1
      
        toggle = not toggle
        if toggle:
            dis_input = np.concatenate((model_gen.predict(batch_grey_noise), batch_grey), axis=3)
            dis_output = np.zeros((batch_size, 1))
        else:
            dis_input = np.concatenate((batch_color, batch_grey), axis=3)
            dis_output = np.random.uniform(low=0.8, high=1.1, size=batch_size)
      
        dis_results = model_dis.train_on_batch(dis_input, dis_output)
      
        gen_input = batch_grey_noise
        dis_output = np.random.uniform(low=0.8, high=1.1, size=batch_size)
        gen_output = batch_color
        gan_res = model_gan.train_on_batch(gen_input, [dis_output, gen_output])
              
        progbar.add(batch_size,
                    values=[("D loss", dis_results),
                            ("G total loss", gan_res[0]),
                            ("G loss", gan_res[1]),
                            ("G L1", gan_res[2]),
                            ("dis acc", gan_res[4]),
                            ("pacc", gan_res[7]),
                            ("acc", gan_res[8]),
                            ("mse", gan_res[9]),
                            ("mae", gan_res[10])])
      
        summary = utils.create_summary_batch(dis_results, gan_res)
        writer.add_summary(summary, global_batch_counter)
        global_batch_counter += 1
           
    # evaluate on test dataset
    if (data_limit == -1):
        data_grey_test_noise = utils.add_noise(data_grey_test)
        data_color_test = data_test[:, :, :, 1:]
        
        if e % 10 == 0:
            ev = model_gan.evaluate(data_grey_test_noise, [np.ones((data_grey_test_noise.shape[0], 1)), data_color_test])
            ev = np.round(np.array(ev), 4)
            summary = utils.create_summary_epoch(ev)
            writer.add_summary(summary, e)
         
        if e % 10 == 0:
            image_values = []
            for i in range (0, 50):
                grey = data_grey[i]
                grey_noise = data_grey_noise[i]
                color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
                image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
                image_value = utils.create_image_summary(image_pred, i)
                image_values.append(image_value)
            summary = tf.Summary(value = image_values)
            writer.add_summary(summary, e)
             
            test_image_values = []
            for i in range (0, 50):
                grey = data_grey_test[i]
                grey_noise = data_grey_test_noise[i]
                color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
                image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
                test_image_value = utils.create_image_summary(image_pred, i, "_test")
                test_image_values.append(test_image_value)
            summary = tf.Summary(value = test_image_values)
            writer.add_summary(summary, e)
            
        if e % 10 == 0:
            rgb_pred_values = []
            for i in range (data_test.shape[0]):
                grey = data_grey_test[i]
                grey_noise = data_grey_test_noise[i]
                color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
                image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
                rgb_pred_values.append(image_pred)
            rgb_pred_values = np.array(rgb_pred_values)
            class_acc = evaluator.evaluate(rgb_pred_values, lables_test)    
            colorfulness = calculate_colorfulness(rgb_pred_values)
            summary = utils.create_summary_evaluation(class_acc, colorfulness)
            writer.add_summary(summary, e)
            utils.save_weights(results_dir, model_gen, model_dis, model_gan, str(e))
    # evaluate on valid data        
    else:
        data_grey_valid = data_valid[:, :, :, :1]
        data_grey_valid_noise = utils.add_noise(data_grey_valid)
        data_color_valid = data_valid[:, :, :, 1:]
        
        if e % 10 == 0:
            ev = model_gan.evaluate(data_grey_valid_noise, [np.ones((data_grey_valid_noise.shape[0], 1)), data_color_valid])
            ev = np.round(np.array(ev), 4)
            summary = utils.create_summary_epoch(ev)
            writer.add_summary(summary, e)
         
        if e % 10 == 0:
            image_values = []
            for i in range (0, 50):
                grey = data_grey[i]
                grey_noise = data_grey_noise[i]
                color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
                image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
                image_value = utils.create_image_summary(image_pred, i)
                image_values.append(image_value)
            summary = tf.Summary(value = image_values)
            writer.add_summary(summary, e)
             
            valid_image_values = []
            for i in range (0, 50):
                grey = data_grey_valid[i]
                grey_noise = data_grey_valid_noise[i]
                color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
                image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
                valid_image_value = utils.create_image_summary(image_pred, i, "_valid")
                valid_image_values.append(valid_image_value)
            summary = tf.Summary(value = valid_image_values)
            writer.add_summary(summary, e)
            
        if e % 10 == 0:
            rgb_pred_values = []
            for i in range (data_valid.shape[0]):
                grey = data_grey_valid[i]
                grey_noise = data_grey_valid_noise[i]
                color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
                image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
                rgb_pred_values.append(image_pred)
            rgb_pred_values = np.array(rgb_pred_values)
            class_acc = evaluator.evaluate(rgb_pred_values, lables_valid)    
            colorfulness = calculate_colorfulness(rgb_pred_values)
            summary = utils.create_summary_evaluation(class_acc, colorfulness)
            writer.add_summary(summary, e)
            utils.save_weights(results_dir, model_gen, model_dis, model_gan, str(e))
            
