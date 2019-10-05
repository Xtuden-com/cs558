from convolution import convulve2d
import clip 
import numpy

def hessianDeterminant(image):
    #sobel filters
    sobel_x = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = numpy.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # first order gradients
    image_x = convulve2d(image,sobel_x)
    image_y = convulve2d(image,sobel_y)
    # second order gradients
    image_xx = convulve2d(image_x,sobel_x)
    image_xy = convulve2d(image_x,sobel_y)
    image_yy = convulve2d(image_y,sobel_y)
    ret_image = image_xx * image_yy - image_xy ** 2
    return ret_image

def threshold(image, value):
    ret_image = numpy.zeros(image.shape)
    x_shape, y_shape = image.shape
    # remove if below threshold value
    for row in range(x_shape):
        for col in range(y_shape):
            if image[row][col] > value:
                ret_image[row][col] = image[row][col]
    return ret_image

def nonMaxSuppression(image):
    image = clip.padImage(image,1)
    ret_image = numpy.zeros(image.shape)
    x_shape, y_shape = image.shape
    # iterate through entire image
    for i in range(1,x_shape-1):
        for j in range(1,y_shape-1):
            value = image[i][j]
            image[i][j] = 0
            # matrix around pixel
            temp_matrix = image[i-1:i+2,j-1:j+2]
            # if the max value is less than the center add it
            if numpy.amax(temp_matrix) < value:
                ret_image[i][j] = value
            # reset image value
            image[i][j] = value
    ret_image = clip.clipImage(ret_image,1)
    image = clip.clipImage(image,1)
    return ret_image