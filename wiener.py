import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import scipy.signal as sig
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to image')
args = vars(ap.parse_args())

img = cv2.imread(args['image'])
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

noise = np.random.standard_normal(img.shape) * 10
noise_img = img + noise
img = noise_img


def correlate(in1, in2):
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)
    index_inv2 = (slice(None, None, -1),) * in2.ndim
    conj_2 = in2[index_inv2].conj()
    out = sig.convolve(in1, conj_2, mode='same')
    return out

def mean(inp, size):
    kernel = np.ones(size) / np.product(size, axis=0)
    out = sig.convolve2d(inp, kernel, 'same')
    return out
    
def wiener(img, size=None, noise=None):
    img = np.asarray(img)
    if size is None: 
        size = [3] * img.ndim
    size = np.asarray(size)
       
    local_mean = correlate(img, np.ones(size)) / np.product(size, axis=0)  
    local_var = correlate(img**2, np.ones(size)) / np.product(size, axis=0)

    if noise is None:
        noise = np.mean(np.ravel(local_var), axis=0)
    
    out = img - local_mean

    out *= (1-noise/(local_var+1e-8))
    out += local_mean
    out_ = np.where(local_var<noise, local_mean, out)
    return np.uint8(out_)

a = cv2.GaussianBlur(img, (3, 3), 1)
b = wiener(img)

plt.figure(figsize=(20, 20))

plt.subplot(331)
plt.xlabel('Ảnh đầu vào')
plt.imshow(img, cmap=plt.cm.gray)

plt.subplot(332)
plt.xlabel('Bộ lọc Gauss')
plt.imshow(a, cmap=plt.cm.gray)

plt.subplot(333)
plt.xlabel('Bộ lọc Wiener')
plt.imshow(b, cmap=plt.cm.gray)

plt.show()
