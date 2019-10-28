import numpy
import clip
import math

def gradientInfo(x_gradient, y_gradient, threshold):
    x_size, y_size = x_gradient.shape
    magnitude = numpy.zeros(x_gradient.shape)
    direction = numpy.zeros(x_gradient.shape)
    for i in range(x_size):
        for j in range(y_size):
            distance =  (x_gradient[i][j] ** 2 + y_gradient[i][j] ** 2) ** .5
            # threshold
            if (distance > threshold):
                magnitude[i][j] = distance
    direction = numpy.arctan2(y_gradient,x_gradient) * 180 / numpy.pi
    return magnitude,direction

# check if your center is the max value
def maxValue(ret,magnitude,row,col,x,y):
    if magnitude[row+x][col+y] > magnitude[row][col] or magnitude[row-x][col-y] > magnitude[row][col]:
        ret[row][col] = 0
    else:
        ret[row][col] = magnitude[row][col]

def nonMaxSuppression(magnitude,direction):
    #pad by 1 so we cant check a 3x3 square for the max
    clip.padImage(magnitude,1)
    x_size, y_size = magnitude.shape   
    ret = numpy.zeros(magnitude.shape)
    for row in range(1,x_size-1):
        for col in range(1,y_size-1):
            c_direction = direction[row][col]
            # horizontal
            if c_direction > -22.5 and c_direction <= 22.5 or \
            c_direction > 157.5 and c_direction <= -157.5:
                maxValue(ret,magnitude,row,col,1,0)
            # top right and bottom left
            elif c_direction > 22.5 and c_direction <= 67.5 or \
            c_direction > -157.5 and c_direction <= -112.5:
                maxValue(ret,magnitude,row,col,1,1)
            # vertical
            elif c_direction > 67.5 and c_direction < 112.5 or \
            c_direction > -112.5 and c_direction < -67.5:
                maxValue(ret,magnitude,row,col,1,0)
            else:
                maxValue(ret,magnitude,row,col,1,-1)
    clip.clipImage(ret,1)
    clip.clipImage(magnitude,1)
    return ret
