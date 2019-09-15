import imageio 
import sys
import matplotlib.pyplot as plt
from clip import padImage, clipImage
import convolution
import numpy

if __name__ == "__main__":
    image = imageio.imread(sys.argv[1])
    print(image.shape)
    image = padImage(image,9)
    print(image.shape)
    sobel = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T
    #image = convolution.convulve2d(image,sobel)
    image = convolution.convulveGaussian(image,3)
    image = clipImage(image,9)
    print(image.shape)
    plt.imshow(image,cmap=plt.get_cmap(name="gray"))
    plt.show()
    