import numpy
import clip 
import matplotlib.pyplot as plt
import math
import imageio

def hough(image, x_buckets, y_buckets):
    H_accumulator = numpy.array([[0 for _ in range(y_buckets)] for _ in range(x_buckets)])
    theta_intervals = 180 / (y_buckets - 1) 
    max_pixel_value = numpy.amax(image)
    # maximum distance from 0,0
    max_rho = (image.shape[0] ** 2 + image.shape[1] ** 2) ** (1/2)
    # * 2 since we need to add all values by x_buckets//2 
    rho_intervals = max_rho * 2 / (x_buckets - 1) 
    ret_image = numpy.copy(image)
    image = ret_image
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
    imageio.imsave('results/h_accumulator.png',H_accumulator)
    #only for visualization to plot on top of the original image
    road = imageio.imread('road.png')
    for i in range(4):
        extractLine(H_accumulator, image, theta_intervals, rho_intervals, points, max_pixel_value, road)
    imageio.imsave('results/hough_over_road.png',road)
    return image

# draw the line
def extractLine(H_accumulator, image, theta_intervals, rho_intervals, points, max_pixel_value, road):
    max_value = -1
    max_x = 0 
    max_y = 0
    # find the max value of H_accumulator
    for i in range(H_accumulator.shape[0]):
        for j in range(H_accumulator.shape[1]):
            if H_accumulator[i][j] > max_value:
                max_value = H_accumulator[i][j]
                max_x = i
                max_y = j
    theta = max_y * theta_intervals * numpy.pi / 180 
    rho = (max_x - H_accumulator.shape[0]//2) * rho_intervals
    # draw the line given theta and rho
    drawPerpendicular(image,rho,theta, points,max_pixel_value, road)
    # get rid of the max value 
    H_accumulator[max_x][max_y] = 0
 
def drawPerpendicular(image,rho,theta, points,max_value, road):
    # the point in spatial domain
    x = rho * numpy.cos(theta)
    y = rho * numpy.sin(theta)
    sin = numpy.sin(theta)
    cos = numpy.cos(theta)
    # if the cos is small enough just say it is 0 slope
    if cos != 0 and cos > .00001:
        slope = -sin / cos
        intercept = rho / cos
        for i in range(image.shape[0]):
            val = intercept + slope * i
            if val >= 0 and val <= image.shape[1]:
                image[i][int(val)] = max_value
                road[i][int(val)] = 255
        # put a 3x3 box around a point on the line if it is within bounds
        for point in points:
            if point[0]>=1 and point[0]<=image.shape[1]-1 and point[1]>=1 and image.shape[0]-1:
                val = int(intercept + slope * point[0])
                if abs(val-point[1]) <= 2:
                    image[-1+point[0]:point[0]+2,-1+point[1]:point[1]+2] = max_value
                    road[-1+point[0]:point[0]+2,-1+point[1]:point[1]+2] = 255
    #horizontal line
    else:
        for i in range(image.shape[1]):
            if int(x) >= 0 and int(x) <= image.shape[1]:
                image[int(x)][i] = max_value
                road[int(x)][i] = 255
        # put a 3x3 box around a point on the line if it is within bounds
        for point in points:
            if point[0]>=1 and point[0]< image.shape[1]-1 and point[1]>=1 and point[1] < image.shape[0]-1:
                if abs(point[0] - int(x)) <=2:
                    image[-1+point[0]:point[0]+2,-1+point[1]:point[1]+2] = max_value
                    road[-1+point[0]:point[0]+2,-1+point[1]:point[1]+2] = 255