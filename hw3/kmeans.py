import numpy
import sys
from random import randint

def euclideanDistance(pixel1,pixel2):
    R = (pixel1[0] - pixel212[0]) ** 2
    G = (pixel1[1] - pixel2[1]) ** 2
    B = (pixel1[2] - pixel2[2]) ** 2
    distance = (R + G + B) ** (1/2)
    return distance

def clusterAverage(cluster):
    total = [0,0,0]
    length = len(cluster)
    # in case one of the clusters is empty
    if length != 0:
        for i in range(length):
            total[0] += cluster[i][0]
            total[1] += cluster[i][1]
            total[2] += cluster[i][2]
        total[0] = total[0] / length
        total[1] = total[1] / length
        total[2] = total[2] / length
    return total

def updated(centers, nextSet):
    for i in range(len(centers)):
        if centers[i] != nextSet[i]:
            return True
    return False

def findCenters(k,image):
    centers = []
    clusters_color = None
    clusters_index = None
    previousSet = None
    xsize, ysize, _ = image.shape
    # randomly find k centers
    for _ in range(k):
        xrandom = randint(0,xsize-1)
        yrandom = randint(0,ysize-1)
        centers.append(image[xrandom][yrandom])
    # if it is the first iteration or the centers have not updated 
    while not previousSet or updated(centers,previousSet):
        clusters_color = [[] for _ in range(k)]
        clusters_index = [[] for _ in range(k)]
        for i in range(xsize):
            for j in range(ysize):
                pixel = image[i][j]
                min_index = 0
                min_val = sys.maxsize
                # find the center the pixel is closest to
                for index in range(k):
                    distance = euclideanDistance(pixel,centers[index])
                    if distance < min_val:
                        min_val = distance
                        min_index = index
                clusters_color[min_index].append(pixel)
                clusters_index[min_index].append([i,j])
        previousSet = centers
        for i in range(k):
            centers[i] = clusterAverage(clusters_color[i])
    return centers, clusters_index
    
def kmeans(k,image):
    centers, clusters_index = findCenters(k,image)
    ret = numpy.zeros(image.shape)
    for i in range(k):
        for pixel_loc in clusters_index[i]:
            # divide by 255 because plt shows 0 to 255
            ret[pixel_loc[0]][pixel_loc[1]][0] = centers[i][0] / 255
            ret[pixel_loc[0]][pixel_loc[1]][1] = centers[i][1] / 255
            ret[pixel_loc[0]][pixel_loc[1]][2] = centers[i][2] / 255
    return ret