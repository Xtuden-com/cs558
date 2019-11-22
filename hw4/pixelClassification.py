import numpy 
import imageio
import sys 
import glob
import matplotlib.pyplot as plt
import heapq
from sklearn.cluster import KMeans

def seperateSets(trainingImage,mask):
    sky = []
    nonSky = []
    print(mask.shape)
    print(trainingImage.shape)

if __name__ == "__main__":
    mask = imageio.imsave("data/sky/sky_train_gimp.jpg")
    trainingImage = imageio.imsave("data/sky/sky_train.jpg")
    seperateSets(trainingImage,mask)
