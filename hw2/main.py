import imageio
import matplotlib.pyplot as plt
import hessian
import numpy
from convolution import convulveGaussian
from ransac import ransac

if __name__ == "__main__":
    image = imageio.imread('road.png')
    # smoothing before any operation is done
    image = convulveGaussian(image,1)
    # hessian determinant
    image = hessian.hessianDeterminant(image)
    plt.imshow(image, cmap='gray')
    plt.show()
    # threshold the values chose 30 as that seems to remove the most gray points
    image = hessian.threshold(image,30)
    # perform non max suprression on 3x3 surrounding pixels
    image = hessian.nonMaxSuppression(image)
    # ransac on threshold 2 and minimum of 16 inliers
    ransac_image = ransac(image, 2, 16)
    
    plt.imshow(ransac_image, cmap='gray')
    plt.show()