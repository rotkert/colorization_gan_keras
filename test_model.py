import numpy as np
from model import create_models
from utils import show_yuv
from dataset import load_cifar10_test_data
 
data_test_yuv, data_test_rgb = load_cifar10_test_data(outType='YUV')
data_test_yuv = data_test_yuv * 255
data_test_y = data_test_yuv[:, :, :, :1]

model_gen, model_dis, model_gan = create_models(
    input_shape_gen=(32,32,1),
    input_shape_dis=(32,32,3),
    output_channels=2,
    lr=0.0001,
    momentum=0.5,
    loss_weights=[1, 10])
    
model_gen.load_weights("F:\\magisterka\\neural_result\\run_01-03-2018\\wagi4\\weights_cifar10_yuv_gen.hdf5")
    
for i in range(0, 5000):
    print(i)
    y = data_test_y[i]
    yuv_original = data_test_yuv[i]
    uv_pred = np.array(model_gen.predict(y[None, :, :, :]))[0]
    yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
    show_yuv(yuv_original / 255, yuv_pred / 255)