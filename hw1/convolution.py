import numpy
import functools

# Convulution function for general odd dimensional kernels
def convulve2d(image, kernel):
    x_size, y_size = image.shape
    ret_image = numpy.zeros(image.shape)
    padding_x = kernel.shape[0]//2
    padding_y = kernel.shape[1]//2
    # loop through the actual image without padding
    for row in range(padding_x,x_size-padding_x):
        for column in range(padding_y,y_size-padding_y):
            total = 0
            # loop through the kernel itself
            for i in range(-1 *  padding_x, padding_x + 1):
                for j in range(-1 * padding_y , padding_y + 1):
                    # make kernel[-1,-1] multiplied by image[1,1]
                    total += kernel[padding_x + i][padding_y + j] * image[row - i][column - j]
            ret_image[row][column] = total
    return ret_image

# Used to verify sum is 1
def sum(x1,x2): return x1 + x2

def convulveGaussian(image,std_deviation):
    # gaussian of 3 std deviations of the mean
    padding = std_deviation * 3
    # 1d gaussian
    gaussian = [(std_deviation ** -1) * (2 * numpy.pi) ** (-1/2) * numpy.exp((-1/2) * (x/std_deviation)**2) \
    for x in range(-1 * padding, padding+1)]
    # verify Gaussian size is 1
    """
    total = 0
    matrix = numpy.outer(gaussian,gaussian)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            total += matrix[i][j]
            print(matrix[i][j], " ",end='')
        print('')
    print(total)
    """
    #matrix = numpy.outer(gaussian,gaussian)
    #image = convulve2d(image,matrix)
    # Convulve 1d twice
    image = convulve1d(image,gaussian)
    image = convulve1d(image.T,gaussian).T
    return image

# 1d convolution
def convulve1d(image, linear_filter):
    x_size , y_size = image.shape
    ret_image = numpy.zeros(image.shape)
    padding = len(linear_filter) // 2 
    # iterate each pixel
    for row in range(padding,x_size-padding):
        for column in range(padding,y_size-padding):
            total = 0 
            # map the padding
            for i in range(-1 * padding, padding + 1):
                total += linear_filter[i + padding] * image[row][column + i]
            ret_image[row][column] = total
    return ret_image