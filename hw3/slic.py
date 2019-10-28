import numpy
import matplotlib.pyplot as plt
import gradient
import convolution
import sys

def initialCenters(image):
    centers = []
    xsize, ysize, _ = image.shape
    xblocks = xsize // 50 
    yblocks = ysize // 50
    for i in range(xblocks):
        for j in range(yblocks):
            x = 25 + i * 50
            y = 25 + j * 50
            centers.append([x,y])
    return centers

def getRGBGradient(image):
    colorChannels = [image[:,:,0],image[:,:,1],image[:,:,2]]
    sobelx = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely = numpy.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    magnitudeArray = []
    for image in colorChannels:
        xgradient = convolution.convulve2d(image,sobelx)
        ygradient = convolution.convulve2d(image,sobely)
        currentMag , _ = gradient.gradientInfo(xgradient,ygradient,0)
        magnitudeArray.append(currentMag)
    magnitude = (magnitudeArray[0] ** 2 + magnitudeArray[1] ** 2 + magnitudeArray[2] ** 2) ** (1/2)
    return magnitude

def findMinIndex(chunk):
    minValue = numpy.min(chunk)
    if chunk[1,1] == minValue:
        return [0,0]
    for i in range(-1,2):
        for j in range(-1,2):
            if chunk[i+1][j+1] == minValue:
                return [i,j]
    return [0,0]

def localShift(centers,magnitude):
    for i in range(len(centers)):
        [x,y] = centers[i]
        if x!=0 and x!=len(magnitude) and y!=0 and y!=len(magnitude[0]):
            chunk = magnitude[x-1:x+2,y-1:y+2]
            [shiftx, shifty] = findMinIndex(chunk)
            centers[i] = [x + shiftx, y + shifty]
    return centers

def euclideanDistance(vector1,vector2):
    total = 0
    for i in range(len(vector1)):
        total += (vector1[i] - vector2[i]) ** 2
    return total ** (1/2)

def getClusterAverage(clustersPosition, clustersColor):
    position = [0,0]
    color = [0,0,0]
    length = len(clustersPosition)
    if length != 0:
        for i in range(len(clustersPosition)):
            pixelIndex = clustersPosition[i]
            position[0] += pixelIndex[0]
            position[1] += pixelIndex[1]
            pixelColor = clustersColor[i]
            color[0] += pixelColor[0]
            color[1] += pixelColor[1]
            color[2] += pixelColor[2]
        position[0] //= length
        position[1] //= length
        color[0] /= length
        color[1] /= length
        color[2] /= length
    return position, color

def updateCentroids(centers,image):
    xsize, ysize, _ = image.shape
    clustersPosition = [[] for _ in range(len(centers))]
    clustersColor = [[] for _ in range(len(centers))]
    colors = [[] for _ in range(len(centers))]
    for i in range(xsize):
        for j in range(ysize):
            pixelCoordinates = [i,j]
            pixel = image[i,j]
            minValue = sys.maxsize
            minIndex = 0
            vector1 = [i/2,j/2,pixel[0],pixel[1],pixel[2]]
            for k in range(len(centers)):
                [x,y] = centers[k]
                if ((x-i) ** 2 + (y-j) ** 2) ** (1/2) <= 71:
                    nextPixel = image[x,y]
                    vector2 = [x/2,y/2,nextPixel[0],nextPixel[1],nextPixel[2]]
                    distance = euclideanDistance(vector1,vector2)
                    if distance < minValue:
                        minValue = distance
                        minIndex = k
            clustersPosition[minIndex].append(pixelCoordinates)
            clustersColor[minIndex].append(pixel)
    for i in range(len(clustersPosition)):
        centers[i], colors[i] = getClusterAverage(clustersPosition[i], clustersColor[i])
    return centers,  colors, clustersPosition

def converge(centers,previousCenter):
    for i in range(len(centers)):
        pixel1 = centers[i]
        pixel2 = previousCenter[i]
        if pixel1[0]!=pixel2[0] or pixel1[1]!=pixel2[1]:
            return False
    return True

def colorCenters(image,centers):
    xsize, ysize, _ = image.shape
    for center in centers:
        center[0] = int(center[0])
        center[1] = int(center[1])
        if center[0]!=0 and center[0]!=xsize and center[1]!= 0 and center[1]!=ysize:
            image[center[0]-1:center[0]+2,center[1]-1:center[1]+2] = [0,0,0]
    image = image / 255
    plt.imshow(image)
    plt.show()

def fillClusters(clusters, colors, image):
    return None

def drawBorders(image):
    return None

def slic(image):
    centers = initialCenters(image)
    previousCenters = None
    gradientMagnitude = getRGBGradient(image)
    iterations = 0
    clusters = None
    colors = None
    print ('begin')
    while iterations != 3:
        previousCenters = centers.copy()
        centers = localShift(centers,gradientMagnitude)
        centers, colors, clusters  = updateCentroids(centers,image)
        print('iteration: ', iterations)
        if converge(centers,previousCenters):
            break
        iterations += 1
    print('color now')
    colorCenters(image,centers)
    ret = fillClusters(image,clusters,colors)
    return None