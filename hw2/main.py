import imageio
import matplotlib.pyplot as plt
import numpy
from convolution import convulveGaussian
import sys 


import hessian
from ransac import ransac
from hough import hough


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error: python main.py threshold_for_ransac inliers_for_ransac theta_dimension rho_dimension")
        exit(1)
    threshold = int(sys.argv[1])
    inliers = int(sys.argv[2])
    theta_dimension = int(sys.argv[3])
    rho_dimension = int(sys.argv[4])
    image = imageio.imread('road.png')
    # smoothing before any operation is done
    image = convulveGaussian(image,1)
    # hessian determinant
    image = hessian.hessianDeterminant(image)
    # threshold the values as 40 of 255 if the image was bounded to be 0 to 255 
    theshold_value = numpy.amax(image) * 40/255
    image = hessian.threshold(image,theshold_value)
    imageio.imwrite('results/hessian_with_threshold.png',image)
    # perform non max suprression on 3x3 surrounding pixels
    image = hessian.nonMaxSuppression(image)
    imageio.imwrite('results/hessian_max_suppression.png',image)
    # ransac on threshold 2 and minimum of 16 inliers
    ransac_image = ransac(image, threshold, inliers)
    imageio.imwrite('results/ransac_over_hessian.png',ransac_image)
    plt.imshow(ransac_image,cmap='gray')
    plt.show()
    # hough with parameters size in the x direction and size in y
    hough_image = hough(image, theta_dimension, rho_dimension)
    imageio.imwrite('results/hough_over_hessian.png',hough_image)
    plt.imshow(hough_image, cmap='gray')
    plt.show()  