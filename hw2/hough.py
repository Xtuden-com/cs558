import numpy
import clip 
import matplotlib.pyplot as plt
import math
import imageio

def hough(image, x_buckets, y_buckets):
    H_accumulator = numpy.array([[0 for _ in range(y_buckets)] for _ in range(x_buckets)])
    theta_intervals = 180 / (y_buckets - 1) 
    # maximum distance from 0,0
    max_rho = (image.shape[0] ** 2 + image.shape[1] ** 2) ** (1/2)
    # * 2 since we need to add all values by x_buckets//2 
    rho_intervals = max_rho * 2 / (x_buckets - 1) 
    points = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 0:
                points.append([i,j])
    for point in points:
        for theta_index in range(y_buckets):
            theta = theta_index * theta_intervals * numpy.pi / 180
            # actual rho
            rho = point[1] * numpy.cos(theta) + point[0] * numpy.sin(theta)
            # rho index, center it
            rho_index = rho / rho_intervals + x_buckets // 2
            H_accumulator[int(rho_index)][theta_index] += 1
    for i in range(4):
        extractLine(H_accumulator, image, theta_intervals, rho_intervals, points)
    return image

def extractLine(H_accumulator, image, theta_intervals, rho_intervals, points):
    max_value = -1
    max_x = 0 
    max_y = 0
    for i in range(H_accumulator.shape[0]):
        for j in range(H_accumulator.shape[1]):
            if H_accumulator[i][j] > max_value:
                max_value = H_accumulator[i][j]
                max_x = i
                max_y = j
    theta = max_y * theta_intervals * numpy.pi / 180 
    rho = (max_x - H_accumulator.shape[0]//2) * rho_intervals
    drawPerpendicular(image,rho,theta, points)
    H_accumulator[max_x][max_y] = 0
 
def drawPerpendicular(image,rho,theta, points):
    x = rho * numpy.cos(theta)
    y = rho * numpy.sin(theta)
    sin = numpy.sin(theta)
    if sin != 0 and sin > .000001:
        slope = -numpy.cos(theta) / sin
        intercept = rho / sin
        for i in range(image.shape[1]):
            val = intercept + slope * i
            if val >= 0 and val <= image.shape[0]:
                image[int(val)][i] = 255
    else:
        for i in range(image.shape[0]):
            if int(x) >= 0 and int(x) <= image.shape[1]:
                image[i][int(x)] = 255

if __name__ == "__main__":
    image = imageio.imread('canny.gif')
    print(image.shape)
    x = 11
    image = hough(image,1000,180)
    plt.imshow(image,cmap='gray')
    plt.show()