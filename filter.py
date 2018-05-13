import cv2
import numpy as np
import dataset
from skimage import color
import matplotlib.pyplot as plt

# Create a dummy input image.

images, _ = dataset.load_stl10_test_data()
# images = dataset.preproc_cifar(images)

# for i in range(0, 1000):
image1 = images[0]
image2 = images[7]
image3 = images[74]
image4 = images[81]


image1 = color.rgb2gray(image1)
image1 = color.gray2rgb(image1)

image2 = color.rgb2gray(image2)
image2 = color.gray2rgb(image2)

image3 = color.rgb2gray(image3)
image3 = color.gray2rgb(image3)

image4 = color.rgb2gray(image4)
image4 = color.gray2rgb(image4)


kernel = np.array([[ 0.39917183, -0.02137024 ,-0.09801425,  0.06261232],
 [-0.24629296,  0.00448981,  0.2054418 ,  0.10748953],
 [-0.3489517 ,  0.05455715 , 0.09855421 , 0.16552567],
 [-0.29267117 ,-0.01971953  ,0.12467878 , 0.01183503]])


dst1 = cv2.filter2D(image1, -1, kernel)
dst2 = cv2.filter2D(image2, -1, kernel)
dst3 = cv2.filter2D(image3, -1, kernel)
dst4 = cv2.filter2D(image4, -1, kernel)

dst1 = 1.5 * dst1
dst1 = np.clip(dst1, 0, 1)

dst2 = 1.5 * dst2
dst2 = np.clip(dst2, 0, 1)

dst3 = 1.5 * dst3
dst3 = np.clip(dst3, 0, 1)

dst4 = 1.5 * dst4
dst4 = np.clip(dst4, 0, 1)

fig = plt.figure()

fig.add_subplot(2, 2, 1)
plt.axis('off')
plt.imshow(image1)

fig.add_subplot(2, 2, 2)
plt.axis('off')
plt.imshow(dst1)

fig.add_subplot(2, 2, 3)
plt.axis('off')
plt.imshow(image2)

fig.add_subplot(2, 2, 4)
plt.axis('off')
plt.imshow(dst2)

plt.subplots_adjust(bottom=0.0, left=.1, right=0.9, top=.9, hspace=.05, wspace=0.005)
plt.show()
