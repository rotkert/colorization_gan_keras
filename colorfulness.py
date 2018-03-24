import numpy as np
import dataset
import utils
from skimage import color
import matplotlib.pyplot as plt


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
    
    
def test(data):
    a = []
    for i in range (0, data.shape[0]):
        colorf = calculate_image_colorfullness(data[i])
        a.append(colorf)
    
    return np.mean(np.array(a))