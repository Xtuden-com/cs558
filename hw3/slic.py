import numpy
import matplotlib.pyplot as plt
import gradient
import convolution

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
    minValue = numpy.matrix.min(chunk)
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
            [shiftx, shifty] = findMinIndex[chunk]
            centers[i] = [x + shiftx, y + shifty]
    return centers

def slic(image):
    centers = initialCenters(image)
    previousCenters = None
    gradientMagnitude = getRGBGradient(image)
    iterations = 0 
    #while iterations != 3 and (not previousCenter or not converge(centers,previousCenters)):
    
    return None
