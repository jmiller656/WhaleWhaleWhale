import joblib
import os
from makeLMfilters import F as filters
import scipy.misc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import time

km = joblib.load("../resources/km_whale.joblib")

def segmentImg(img):
    #Convert to grayscale if necessary
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Apply each filter to the image and store it in X
    X = np.array([cv2.filter2D(img, -1, filters[:,:,i]) for i in range(filters.shape[2])])
    X = np.abs(X)

    #Doing this mostly for my own sanity. Easier to think of as [width, height ,channels]
    X = X.transpose((1,2,0))

    #Flatten our filtered array to form a list of voxels
    voxels =X.reshape((-1,X.shape[-1]))

    #Cluster on the voxels using sklearn kmeans (261 looked like a nice number)
    #kmeans = KMeans(n_clusters=k, random_state=261).fit(voxels)
    #joblib.dump(kmeans, "km_whale.joblib")
    #Get kmenas labels for each voxel
    preds = km.predict(voxels)

    #Get new image where each pixel contains the label for that pixel
    labeled = preds.reshape((img.shape[0], img.shape[1]))
    return labeled#segmented_image(labeled)

"""
Cheap way of visualizing
segmented image
"""
def segmented_image(labels):
    colors = {}
    colors[0] = [255,0,0]
    colors[1] = [0,255,0]
    colors[2] = [0,0,255]
    colors[3] = [255,255,0]
    colors[4] = [0,255,255]
    colors[5] = [255,0,255]
    colors[6] = [0,0,0]
    colors[7] = [255,255,255]
    tmp = np.zeros([labels.shape[0], labels.shape[1], 3])
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            tmp[x,y] = colors[labels[x,y]]
    return tmp

def mul_img(img, lab):
    img = np.uint8(img)
    lab = np.uint8(lab)
    lab = np.abs(lab-1)
    img[:,:,0] *= lab
    img[:,:,1] *= lab
    img[:,:,2] *= lab
    return img
def main():
    fn = "../train/"
    files = os.listdir(fn)
    for file in files:
        print(file)
        im = scipy.misc.imread(fn+file)
        tmp = segmentImg(im)
        if len(im.shape) >2:
            plt.imshow(np.uint8(im))
        else:
            plt.imshow(im, cmap='gray')
        plt.show()
        if len(im.shape) > 2:
            plt.imshow(mul_img(im,tmp)[:,:,::-1])
        else:
            plt.imshow(im*tmp, cmap='gray')
        plt.show()
        time.sleep(2)

if __name__ == '__main__':
    main()
