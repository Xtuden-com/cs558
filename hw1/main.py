import imageio 
import sys
import matplotlib.pyplot as plt
import convolution
import numpy
import gradient


#I pledge my honor that I have abided by the Stevens Honor System. VL
if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage: python main.py file/path/to/image std_deviation")
        exit(1)
    image = imageio.imread(sys.argv[1])
    std_deviation = int(sys.argv[2])
    sobel = numpy.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    image = convolution.convulveGaussian(image,std_deviation)
    x_image = convolution.convulve2d(image,sobel)   
    y_image = convolution.convulve2d(image,sobel.T) 
    magnitude, direction = gradient.gradientInfo(x_image,y_image,90)
    image = gradient.nonMaxSuppression(magnitude,direction)
    plt.imshow(image,cmap=plt.get_cmap(name="gray"))
    plt.show()        