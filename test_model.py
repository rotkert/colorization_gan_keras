import numpy as np
import utils
import model_transp
from dataset import load_train_data, load_test_data
from utils_visualisation import show_yuv
from keras.models import load_model
import colorfulness

LEARNING_RATE = 0.0001
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 100
DATASET = "cifar10"
DATA_LIMIT = -1
COLORSPACE = "YUV" 

data_yuv,  mean = load_train_data(dataset = DATASET, data_limit = DATA_LIMIT, colorspace = COLORSPACE)
data_test_yuv, _ = load_test_data(dataset = DATASET, colorspace = COLORSPACE, mean = mean)
data_test_y = data_test_yuv[:, :, :, :1]
data_test_y_noise = utils.add_noise(data_test_y)

model_gen, model_dis, model_gan = model_transp.create_models(
        input_shape_gen = (32, 32, 4),
        input_shape_dis = (32, 32, 3),
        output_channels=2,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        loss_weights=[LAMBDA1, LAMBDA2])

model_gen.load_weights("C:\\Users\\Miko\\Desktop\\test\\weigths_epoch_275\\weights_gen.h5")
     
for i in range(0, 5000):
    print(i)
    y = data_test_y[i]
    y_noise = data_test_y_noise[i]
    yuv_original = data_test_yuv[i]
    uv_original = yuv_original[:, :, 1:]
    uv_pred = np.array(model_gen.predict(y_noise[None, :, :, :]))[0]
    rgb_pred = utils.process_after_predicted(uv_original, y, mean)
    print(colorfulness.calculate_image_colorfullness(rgb_pred))
    show_yuv(yuv_original, rgb_pred, mean)