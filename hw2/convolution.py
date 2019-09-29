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
    # form the 2d matrix
    gaussian = numpy.outer(gaussian,gaussian)
    total = sum(map(sum,gaussian))
    # normalize to ensure 1 as the total of the gaussian
    gaussian /= total
    image = convulve2d(image,gaussian)
    return image

