import numpy

# Pad the image with 0's i.e. clip filter
# pad size is the padding on each side
def padImage(image, pad_size):
    shape = image.shape
    x_size = shape[0] + pad_size * 2
    y_size = shape[1] + pad_size * 2
    ret_image = numpy.zeros((x_size,y_size))
    # fill in the image into the padding
    for row in range(shape[0]):
        ret_image[row+pad_size][pad_size:y_size-pad_size] = image[row]
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