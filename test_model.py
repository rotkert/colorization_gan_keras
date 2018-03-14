import numpy as np
from model_max_pool import create_models
from utils_visualisation import show_yuv
from dataset import load_train_data
 
data_test_yuv, _, _, _ = load_train_data("cifar100", 1000, "YUV")
data_test_yuv = data_test_yuv * 255
data_test_y = data_test_yuv[:, :, :, :1]

model_gen, model_dis, model_gan = create_models(
    input_shape_gen=(32,32,1),
    input_shape_dis=(32,32,3),
    output_channels=2,
    lr=0.0001,
    momentum=0.5,
    loss_weights=[1, 10])
    
model_gen.load_weights("F:\\magisterka\\neural_result\\normal\\ganl1_max_pool_cifar10_YUV_bs128_run-2018-03-08_2044\weigths_epoch_80\\model_gen.h5")
model_dis.load_weights("F:\\magisterka\\neural_result\\normal\\ganl1_max_pool_cifar10_YUV_bs128_run-2018-03-08_2044\weigths_epoch_80\\model_dis.h5")
model_gan.load_weights("F:\\magisterka\\neural_result\\normal\\ganl1_max_pool_cifar10_YUV_bs128_run-2018-03-08_2044\weigths_epoch_80\\model_gan.h5")
    
for i in range(0, 5000):
    print(i)
    y = data_test_y[i]
    yuv_original = data_test_yuv[i]
    uv_pred = np.array(model_gen.predict(y[None, :, :, :]))[0]
    yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
    show_yuv(yuv_original / 255, yuv_pred / 255)