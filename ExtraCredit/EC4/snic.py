import numpy
import matplotlib.pyplot as plt
import gradient
import convolution
import sys
import heapq

def initialCenters(image):
    # centers are 50 apart starting at 25,25
    colors = []
    centers = []
    xsize, ysize, _ = image.shape
    xblocks = xsize // 50 
    yblocks = ysize // 50
    for i in range(xblocks):
        for j in range(yblocks):
            x = 25 + i * 50
            y = 25 + j * 50
            centers.append([x,y])
            colors.append(image[x][y])
    return centers

def drawBorders(image,clusters):
    xsize, ysize, _ = image.shape
    ret = numpy.zeros(image.shape)
    for cluster in clusters:
        for pixel in cluster:
            if pixel[0] < xsize - 1 and pixel[1] < ysize - 1:
                if clusterContains([pixel[0]+1,pixel[1]],[pixel[0],pixel[1]+1],cluster,image):
                    ret[pixel[0],pixel[1]] = image[pixel[0],pixel[1]]
                else:
                    ret[pixel[0],pixel[1]] = [0,0,0]
            else:
                ret[pixel[0],pixel[1]] = image[pixel[0],pixel[1]]
    return ret

# fill in the clusters with color
def fillClusters(clusters, colors, image):
    ret = numpy.zeros(image.shape)
    for i in range(len(colors)):
        for point in clusters[i]:
            ret[point[0],point[1]] = colors[i]
    return ret

def updateCluster(clusters, index, element):
    cluster = clusters[index]
    cluster[5] += 1
    elements = cluster[5]
    cluster[0] = (element[0] + cluster[0]) // elements
    cluster[1] = (element[1] + cluster[1]) // elements
    for i in range(2,5):
        cluster[i] = (cluster[i] + element[i]) / elements 

def distance(a,b,s,m):


def snic(image, compactnessFactor):
    m = compactnessFactor
    heap = []
    centers, colors = initialCenters(image)
    s = (image.shape[0] * image.shape[1] / len(centers)) ** (1/2)
    ret = numpy.zeros(image.shape)
    clusters = [[0,0,0,0,0,0] for i in range(len(centers))]
    for i in range(len(centers)):
        center = centers[i]
        color = colors[i]
        element = (0,i,color[0],color[1],color[2],center[0],center[1])
        heapq.heappush(heap,element)
    while len(heap) != 0:
        target = heapq.heappop(heap)
        distance, k, colorR, colorG, colorB, x, y = target
        if ret[x][y] == 0: 
            ret[x][y] = [colorR,colorG,colorB]
            updateCluster(clusters, k, [x,y,colorR,colorG,colorB])