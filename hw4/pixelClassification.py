import numpy 
import imageio
import sys 
import glob
import matplotlib.pyplot as plt
import heapq
from sklearn.cluster import KMeans

# use the mask to seperate sky and non sky images
def seperateSets(trainingImage,mask):
    sky = []
    nonSky = []
    x_size, y_size, _ = mask.shape
    for i in range(x_size):
        for j in range(y_size):
            pixel = mask[i][j]
            # the mask has indicated a white pixel as the sky
            if pixel[0] == pixel[1] == pixel[2] == 255:
                sky.append(trainingImage[i][j])
            else:
                nonSky.append(trainingImage[i][j])
    return sky,nonSky

# basic l2 norm
def l2norm(vector1,vector2):
    ret = 0
    for i in range(len(vector1)):
        ret += (vector1[i] - vector2[i]) ** 2
    return ret ** (1/2)

# nearest neighbor
def nearestIsSky(pixel,skyCenters,nonSkyCenters):
    skyHeap = []
    nonSkyHeap = []
    # nearest centroid of the sky portion
    for targetCenter in skyCenters:
        distance = l2norm(pixel,targetCenter)
        heapq.heappush(skyHeap,distance)
    # nearest centroid of the non sky portion
    for targetCenter in nonSkyCenters:
        distance = l2norm(pixel,targetCenter)
        heapq.heappush(nonSkyHeap,distance)
    # if the sky portion minimum distance to the pixel is greater than the non sky
    # the pixel is a non sky pixel
    if heapq.heappop(skyHeap) > heapq.heappop(nonSkyHeap):
        return False
    return True

def pixelClassification():
    # sky color is white
    skyColor = [1,1,1]
    mask = imageio.imread("data/sky/sky_train_gimp.jpg")
    trainingImage = imageio.imread("data/sky/sky_train.jpg")
    sky, nonSky = seperateSets(trainingImage,mask)
    # do kmeans on the sky and non sky pixels
    # this kmeans is supplied by sklearn as we were told we can use library functions to do kmeans
    skyCenters = KMeans(10).fit(sky).cluster_centers_
    nonSkyCenters = KMeans(10).fit(nonSky).cluster_centers_
    testImages = glob.glob('data/sky/sky_test?.jpg')
    for path in testImages:
        image = imageio.imread(path)
        x_size, y_size, _ = image.shape
        ret = numpy.zeros(image.shape)
        # iterate through the image and if it is close to a sky center make it white
        for i in range(x_size):
            for j in range(y_size):
                pixel = image[i][j]
                if nearestIsSky(pixel,skyCenters,nonSkyCenters):
                    ret[i][j] = skyColor 
                else:
                    ret[i][j] = numpy.array(pixel) / 255
        plt.imshow(ret)
        plt.show()