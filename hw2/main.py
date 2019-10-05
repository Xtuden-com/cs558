import imageio
import matplotlib.pyplot as plt
import numpy
from convolution import convulveGaussian
import sys 

import hessian
from ransac import ransac
from hough import hough


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error: python main.py inliers_for_ransac theta_dimension rho_dimension")
        exit(1)
    inliers = int(sys.argv[1])
    theta_dimension = int(sys.argv[2])
    rho_dimension = int(sys.argv[3])
    image = imageio.imread('road.png')
    # smoothing before any operation is done
    image = convulveGaussian(image,1)
    # hessian determinant
    image = hessian.hessianDeterminant(image)
    # threshold the values chose 30 as that seems to remove the most gray points
    image = hessian.threshold(image,80)
    # perform non max suprression on 3x3 surrounding pixels
    image = hessian.nonMaxSuppression(image)
    # ransac on threshold 2 and minimum of 16 inliers
    image = imageio.imread('hessian.png')
    ransac_image = ransac(image, 2, inliers)
    imageio.imwrite('results/overlay.png',ransac_image)
    plt.imshow(ransac_image,cmap='gray')
    plt.show()
    # hough with parameters size in the x direction and size in y
    hough_image = hough(image, theta_dimension, rho_dimension)
    imageio.imwrite('results/hough.png',hough_image)
    plt.imshow(hough_image, cmap='gray')
    plt.show()  