import keras 
import utils
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.0005
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 0
model_gen, model_dis, model_gan = utils.create_models('model_transp_lsgan', 32, LEARNING_RATE, MOMENTUM, LAMBDA1, LAMBDA2)
model_gen.load_weights('F:\OneDrive - Politechnika Warszawska\mgr-wyniki\experiments\gan_lsgan\L1_LSGAN_run-2018-04-21_1409\weights_epoch_100\weights_gen.h5')
#model_gen.summary()
model = model_gen
layer = model.get_layer(name='conv1')
weights = layer.get_weights()

#new = np.reshape(weights[0][0], [64,4,4])

# min = np.amin(new)
# new += min
# max = np.amax(new)
# new = new / max
temp = weights[0][0]
new = [0.5 for k in range(1024)]
new = np.array(new)
new = np.reshape(new, [64,4,4])

for i in range(4):
    for j in range(4):
        for k in range(64):
            a = temp[i,j,k]
            new[k, i, j] = a
            print(a)
            print(new[k,i,j])
            
print(new[61])

fig = plt.figure()
for i in range(new.shape[0]):
    fig.add_subplot(8, 8, i + 1)
    plt.axis('off')
    plt.imshow(new[i], cmap='gray')
     
plt.show()