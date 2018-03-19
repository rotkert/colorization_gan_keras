import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy import misc

def show_yuv(yuv_original, yuv_pred, mean):
    yuv_pred[:, :, 0] += mean[0]
    yuv_pred[:, :, 1] += mean[1]
    yuv_pred[:, :, 2] += mean[2]
    
    yuv_original[:, :, 0] += mean[0]
    yuv_original[:, :, 1] += mean[1]
    yuv_original[:, :, 2] += mean[2]
    
    rgb_original = np.clip(color.yuv2rgb(yuv_original), 0, 1)
    rgb_pred = np.clip(np.abs(color.yuv2rgb(yuv_pred)), 0, 1)
    grey = color.rgb2grey(yuv_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()


def show_rgb(rgb_original, rgb_pred):
    grey = color.rgb2grey(rgb_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()


def show_lab(lab_original, lab_pred):
    lab_pred = lab_pred.astype(np.float64)
    rgb_original = np.clip(color.lab2rgb(lab_original), 0, 1)
    rgb_pred = np.clip(np.abs(color.lab2rgb(lab_pred)), 0, 1)
    grey = color.rgb2grey(lab_original)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1).set_title('greyscale')
    plt.axis('off')
    plt.imshow(grey, cmap='gray')

    fig.add_subplot(1, 3, 2).set_title('original')
    plt.axis('off')
    plt.imshow(rgb_original)

    fig.add_subplot(1, 3, 3).set_title('gan')
    plt.axis('off')
    plt.imshow(rgb_pred)

    plt.show()
    
def save_yuv(yuv_pred, batch_no, image_no):
    rgb_pred = np.clip(np.abs(color.yuv2rgb(yuv_pred)), 0, 1)
    misc.imsave("F:\\magisterka\\results\\image_" + str(batch_no) + "_" + str(image_no) + ".png", rgb_pred)