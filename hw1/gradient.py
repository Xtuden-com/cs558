import numpy
import clip

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
                if x_gradient[i][j] == 0:
                    direction[i][j] = 1.57
                else:
                    direction[i][j] = numpy.arctan(y_gradient[i][j]/x_gradient[i][j])
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
            # up and down
            if (direction[row][col] >= -1* numpy.pi/2 and direction[row][col] <= -3 * numpy.pi/8 or \
            direction[row][col] <= numpy.pi/2 and direction[row][col] > 3* numpy.pi / 8):
                maxValue(ret,magnitude,row,col,0,1)
            # bottom right and top left
            if (direction[row][col] > -3 * numpy.pi/8 and direction[row][col] <= -1 * numpy.pi/8):
                maxValue(ret,magnitude,row,col,1,-1)
            # horizontal
            if (direction[row][col] > -1*numpy.pi/8 and direction[row][col] <= numpy.pi/8):
                maxValue(ret,magnitude,row,col,1,0)
            # top right and bottom left
            if (direction[row][col] > numpy.pi/8 and direction[row][col] <= 3*numpy.pi/8):
                maxValue(ret,magnitude,row,col,1,1)
    clip.clipImage(ret,1)
    clip.clipImage(magnitude,1)
    return ret