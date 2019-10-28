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



def slic(image):
    centers = initialCenters(image)
    gradientMagnitude = getRGBGradient(image)
    return None
