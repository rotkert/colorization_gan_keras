import cv2
import numpy as np
import dataset
from skimage import color
import matplotlib.pyplot as plt

# Create a dummy input image.

images, _ = dataset.load_cifar10_test_data()
images = dataset.preproc_cifar(images)

for i in range(60, 120):
    image = images[i]
    
    image = color.rgb2gray(image)
    print(image.shape)
    
    image = color.gray2rgb(image)
    
    kernel = np.array([[ 0.39917183, -0.02137024 ,-0.09801425,  0.06261232],
     [-0.24629296,  0.00448981,  0.2054418 ,  0.10748953],
     [-0.3489517 ,  0.05455715 , 0.09855421 , 0.16552567],
     [-0.29267117 ,-0.01971953  ,0.12467878 , 0.01183503]])
    
    
    dst = cv2.filter2D(image, -1, kernel)
    
    
    
    print(np.max(dst))
    
    dst = np.clip(dst, 0, 1)
    
    fig = plt.figure()
    
    fig.add_subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(images[i])
    
    fig.add_subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(image)

    fig.add_subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(dst)

    plt.show()
