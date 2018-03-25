import numpy as np
import utils
import model_transp
from dataset import load_train_data, load_test_data
from utils_visualisation import show_yuv
from keras.models import load_model
from utils_evaluation import evaluator
from utils_evaluation import calculate_colorfulness

LEARNING_RATE = 0.0001
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 100
DATASET = "cifar10"
DATA_LIMIT = -1
COLORSPACE = "YUV" 

data_yuv,  mean = load_train_data(dataset = DATASET, data_limit = DATA_LIMIT, colorspace = COLORSPACE)
data_test_yuv, _ = load_test_data(dataset = DATASET, colorspace = COLORSPACE, mean = mean)
data_valid_yuv, lables_valid = load_valid_data(dataset = DATASET, colorspace = COLORSPACE, mean = mean, size = 500)
data_test_y = data_test_yuv[:, :, :, :1]
data_test_y_noise = utils.add_noise(data_test_y)

data_valid_y = data_valid_yuv[:, :, :, :1]
data_valid_y_noise = utils.add_noise(data_valid_y)

model_gen, model_dis, model_gan = model_transp.create_models(
        input_shape_gen = (32, 32, 4),
        input_shape_dis = (32, 32, 3),
        output_channels=2,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        loss_weights=[LAMBDA1, LAMBDA2])

model_gen.load_weights("C:\\Users\\Miko\\Desktop\\test\\weigths_epoch_275\\weights_gen.h5")
     
rgb_pred_values = []
for i in range (data_valid_yuv.shape[0]):
    y = data_valid_y[i]
    y_noise = data_valid_y_noise[i]
    uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
    rgb_pred = utils.process_after_predicted(uv_pred, y, mean)
    rgb_pred_values.append(rgb_pred)
rgb_pred_values = np.array(rgb_pred_values)
class_acc = evaluator.evaluate(rgb_pred_values, lables_valid)    
colorfulness = calculate_colorfulness(rgb_pred_values)
print(class_acc)
print(colorfulness)