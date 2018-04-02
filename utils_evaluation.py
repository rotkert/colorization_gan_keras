import keras
import numpy as np
import dataset
import utils
import tensorflow as tf
from keras.models import load_model

class evaluator:
    def __init__(self, dataset):
        if dataset == "cifar10":
            self.model = load_model("F:\\OneDrive - Politechnika Warszawska\\mgr-wyniki\\models\\vgg16_cifar10.h5")
            self.num_labels = 10
            self.mean = 120.707
            self.std = 64.15
        elif dataset == "cifar100":
            self.model = load_model("F:\\OneDrive - Politechnika Warszawska\\mgr-wyniki\\models\\vgg16_cifar100.h5")
            self.num_labels = 100
            self.mean = 121.936
            self.std = 68.389

    def normalize_production(self, x):
        return (x - self.mean) / (self.std + 1e-7)

    def evaluate(self, images, labels):
        images = self.normalize_production(images)
        y = keras.utils.to_categorical(labels, self.num_labels)
        ev = self.model.evaluate(images, y)
        ev = np.round(np.array(ev), 4)
        return ev[1]
        
def calculate_image_colorfullness(image):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    
    rg = np.absolute(r - g)
    yb = np.absolute(0.5 * (r + g) - b)
    
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    
    return stdRoot + (0.3 * meanRoot)
    
def calculate_colorfulness(images):
    all = []
    for i in range (images.shape[0]):
        colorfulness = calculate_image_colorfullness(images[i])
        all.append(colorfulness)
    
    return np.mean(np.array(all))
