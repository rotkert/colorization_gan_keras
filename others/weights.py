import numpy as np
import matplotlib.pyplot as plt
import utils

learning_rate = 0.0005
momentum = 0.5
lambda1 = 1
lambda2 = 0
model_gen, model_dis, model_gan = utils.create_models('model_transp_lsgan', 32, learning_rate, momentum, lambda1, lambda2)
model_gen.load_weights('F:\OneDrive - Politechnika Warszawska\mgr-wyniki\experiments\gan_lsgan\L1_LSGAN_run-2018-04-21_1409\weights_epoch_100\weights_gen.h5')
layer = model_gen.get_layer(name='conv1')
weights = layer.get_weights()

temp = weights[0][0]
new = [0.5 for k in range(1024)]
new = np.array(new)
new = np.reshape(new, [64,4,4])

for i in range(4):
    for j in range(4):
        for k in range(64):
            a = temp[i,j,k]
            new[k, i, j] = a
            

fig = plt.figure()
for i in range(new.shape[0]):
    fig.add_subplot(8, 8, i + 1)
    plt.axis('off')
    plt.imshow(new[i], cmap='gray')
     
plt.show()