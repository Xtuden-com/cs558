import imageio
import matplotlib.pyplot as plt
import numpy
from convolution import convulveGaussian

import hessian
from ransac import ransac
from hough import hough

if __name__ == "__main__":
    """
    image = imageio.imread('road.png')
    # smoothing before any operation is done
    image = convulveGaussian(image,1)
    # hessian determinant
    image = hessian.hessianDeterminant(image)
    plt.imshow(image, cmap='gray')
    plt.show()
    # threshold the values chose 30 as that seems to remove the most gray points
    image = hessian.threshold(image,40)
    # perform non max suprression on 3x3 surrounding pixels
    image = hessian.nonMaxSuppression(image)
    imageio.imwrite('hessian.png',image)
    # ransac on threshold 2 and minimum of 16 inliers
    """
    image = imageio.imread('hessian.png')
    
    image = numpy.array([[0 for _ in range(101)] for _ in range(101)])
    for i in range(0,101, 1):
       image[i][i] = 1
       image[i][100-i] = 1
    
    #image[5][5] = 255
    plt.imshow(image, cmap='gray')
    plt.show()  
    hough_image = hough(image,200,200)
    plt.imshow(hough_image, cmap='gray')
    plt.show()  
    
    """
    #ransac_image = ransac(image, 2, 15)
    image = numpy.array([[0 for _ in range(180)] for _ in range(180)])
    for i in range(0,180, 1):
        image[i][i] = 1
        image[-i][i] = 1
        image[1][i] = 1
        image[i][1] = 1
        
    #image[5][5] = 255
    ransac_image = ransac(image, 1, 15)
    plt.imshow(image,cmap='gray')
    plt.show()
    plt.imshow(ransac_image, cmap='gray')
    plt.show()
    """