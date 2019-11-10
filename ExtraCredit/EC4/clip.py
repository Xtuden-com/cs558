import numpy

# Pad the image with boundaries i.e. clip filter
# pad size is the padding on each side
def padImage(image, pad_size):
    shape = image.shape
    x_size = shape[0] + pad_size * 2
    y_size = shape[1] + pad_size * 2
    ret_image = numpy.zeros((x_size,y_size))
    # for the corners
    for row in range(pad_size):
        for column in range(pad_size):
            ret_image[row][column] = image[0][0]
            ret_image[row+x_size-pad_size][column] = image[shape[0]-1][0]
            ret_image[row+x_size-pad_size][column+y_size-pad_size] = image[shape[0]-1][shape[1]-1]
            ret_image[row][column+y_size-pad_size] = image[0][shape[1]-1]
    # for the horizontal sides
    for row in range(shape[0]):
        ret_image[row+pad_size][0:pad_size] = image[row][0]
        ret_image[row+pad_size][pad_size:y_size-pad_size] = image[row]
        ret_image[row+pad_size][y_size-pad_size:y_size] = image[row][shape[1]-1]
    ret_image = ret_image.T
    # for the vertical sides
    for row in range(shape[1]):
        ret_image[row+pad_size][0:pad_size] = image[0][row]
        ret_image[row+pad_size][x_size-pad_size-1:x_size] = image[shape[0]-1][row]
    ret_image = ret_image.T
    return ret_image

# Remove the padding of 0's from an image
def clipImage(image, clip_size):
    shape = image.shape
    x_size = shape[0] - clip_size * 2
    y_size = shape[1] - clip_size * 2
    ret_image = numpy.zeros((x_size,y_size))
    for row in range(x_size):
        ret_image[row] = image[row + clip_size][clip_size:clip_size+y_size]
    return ret_image