import imageio
import matplotlib.pyplot as plt
import numpy
from convolution import convulveGaussian

import hessian
from ransac import ransac
from hough import hough

if __name__ == "__main__":
    image = imageio.imread('road.png')
    # smoothing before any operation is done
    image = convulveGaussian(image,1)
    # hessian determinant
    image = hessian.hessianDeterminant(image)
    # threshold the values chose 30 as that seems to remove the most gray points
    image = hessian.threshold(image,40)
    # perform non max suprression on 3x3 surrounding pixels
    image = hessian.nonMaxSuppression(image)
    # ransac on threshold 2 and minimum of 16 inliers
    ransac_image = ransac(image, 2, 15)
    plt.imshow(ransac_image,cmap='gray')
    plt.show()
    # hough with parameters size in the x direction and size in y
    hough_image = hough(image,2000,720)
    plt.imshow(hough_image, cmap='gray')
    plt.show()  