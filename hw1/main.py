import imageio 
import sys
import matplotlib.pyplot as plt
from clip import padImage, clipImage
import convolution
import numpy
import gradient

if __name__ == "__main__":
    image = imageio.imread(sys.argv[1])
    image = padImage(image,3)
    sobel = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    image = convolution.convulveGaussian(image,1)
    image = clipImage(image,3)
    image = padImage(image,1)
    x_image = clipImage(convolution.convulve2d(image,sobel),1)
    y_image = clipImage(convolution.convulve2d(image,sobel.T),1)
    magnitude, direction = gradient.gradientInfo(x_image,y_image,50)
    image = gradient.nonMaxSuppression(magnitude,direction,50)
    plt.imshow(image,cmap=plt.get_cmap(name="gray"))
    plt.show()        