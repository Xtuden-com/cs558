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
    x_size, y_size, _ = mask.shape
    for i in range(x_size):
        for j in range(y_size):
            pixel = mask[i][j]
            if pixel[0] == pixel[1] == pixel[2] == 255:
                sky.append(trainingImage[i][j])
            else:
                nonSky.append(trainingImage[i][j])
    return sky,nonSky

def l2norm(vector1,vector2):
    ret = 0
    for i in range(len(vector1)):
        ret += (vector1[i] - vector2[i]) ** 2
    return ret ** (1/2)

def nearestIsSky(pixel,skyCenters,nonSkyCenters):
    skyHeap = []
    nonSkyHeap = []
    for targetCenter in skyCenters:
        distance = l2norm(pixel,targetCenter)
        heapq.heappush(skyHeap,distance)
    for targetCenter in nonSkyCenters:
        distance = l2norm(pixel,targetCenter)
        heapq.heappush(nonSkyHeap,distance)
    if heapq.heappop(skyHeap) > heapq.heappop(nonSkyHeap):
        return False
    return True

def pixelClassification():
    skyColor = [1,1,1]
    mask = imageio.imread("data/sky/sky_train_gimp.jpg")
    trainingImage = imageio.imread("data/sky/sky_train.jpg")
    sky, nonSky = seperateSets(trainingImage,mask)
    skyCenters = KMeans(10).fit(sky).cluster_centers_
    nonSkyCenters = KMeans(10).fit(nonSky).cluster_centers_
    testImages = glob.glob('data/sky/sky_test?.jpg')
    testImages.sort()
    for path in testImages:
        image = imageio.imread(path)
        x_size, y_size, _ = image.shape
        ret = numpy.zeros(image.shape)
        for i in range(x_size):
            for j in range(y_size):
                pixel = image[i][j]
                if nearestIsSky(pixel,skyCenters,nonSkyCenters):
                    ret[i][j] = skyColor 
                else:
                    ret[i][j] = numpy.array(pixel) / 255
        plt.imshow(ret)
        plt.show()

if __name__ == "__main__":
    pixelClassification()