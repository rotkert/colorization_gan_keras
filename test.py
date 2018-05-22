import numpy as np
import matplotlib.pyplot as plt
import utils
import utils_dataset

model = "model_transp_lsgan"
dataset = "cifar100"
colorspace = "LAB"
data_limit = -1

learning_rate = 0.0002
momentum = 0.5
lambda1 = 1
lambda2 = 100

data, mean = utils_dataset.load_train_data(dataset = dataset, data_limit = data_limit, colorspace = colorspace)
data_test, lables_test = utils_dataset.load_test_data(dataset = dataset, colorspace = colorspace, mean = mean)

data_grey = data[:, :, :, :1]
data_color = data[:, :, :, 1:]
 
data_test_grey = data_test[:, :, :, :1]
data_test_color = data_test[:, :, :, 1:]

data_test_grey_noise = utils.add_noise(data_test_grey)

model_gen, model_dis, model_gan = utils.create_models(model, data.shape[1], learning_rate, momentum, lambda1, lambda2)
model_gen.load_weights("weights/weights_gen_cifar100.h5")

for i in range (0, 10000):
    grey = data_test_grey[i]
    grey_noise = data_test_grey_noise[i]
    color_pred = np.array(model_gen.predict(grey_noise[None, :, :, :]))[0]
    image_pred = utils.process_after_predicted(color_pred, grey, mean, colorspace)
    plt.imshow(image_pred)
    plt.show()
