import numpy
import clip
# Convulution function for general odd dimensional kernels
def convulve2d(image, kernel):
    padding_x = kernel.shape[0]//2
    padding_y = kernel.shape[1]//2
    image = clip.padImage(image,padding_x)
    x_size, y_size = image.shape
    ret_image = numpy.copy(image)
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
    image = clip.clipImage(image,padding_x)
    ret_image = clip.clipImage(ret_image,padding_x)
    return ret_image

def convulveGaussian(image,std_deviation):
    # gaussian of 3 std deviations of the mean
    padding = std_deviation * 3
    # 1d gaussian
    gaussian = [(std_deviation ** -1) * (2 * numpy.pi) ** (-1/2) * \
    numpy.exp((-1/2) * (x/std_deviation)**2) \
    for x in range(-1 * padding, padding+1)]
    # Convulve 1d twice
    image = clip.padImage(image,padding)
    image = convulve1d(image,gaussian)
    image = convulve1d(image.T,gaussian).T
    image = clip.clipImage(image,padding)
    return image

# 1d convolution
def convulve1d(image, linear_filter):
    x_size , y_size = image.shape
    ret_image = numpy.copy(image)
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