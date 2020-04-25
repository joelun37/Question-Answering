import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import misc, ndimage, signal
from PIL import Image

def convolution(img, k):
    return np.sum([signal.convolve2d(img[:, :, c], k[:, :, c], mode='valid')
        for c in range(img.shape[2])])

img = plt.imread("/Volumes/750GB-HDD/Documents/_old/Photos/26012008087.jpg")

k = np.array([
    [[ 0,  1, -1], [1, -1, 0], [ 0, 0, 0]],
    [[-1,  0, -1], [1, -1, 0], [ 1, 0, 0]],
    [[ 1, -1,  0], [1,  0, 1], [-1, 0, 1]]])

convoluted_pic = convolution(img, k)

matplotlib.image.imsave("/Volumes/750GB-HDD/Documents/_old/Photos/26012008087_conv.jpg",
                        convoluted_pic)

