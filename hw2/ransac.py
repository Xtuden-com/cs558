import numpy
import sys
import matplotlib.pyplot as plt
import imageio

def ransac(image, threshold, inliers):
    points = []
    x_shape, y_shape = image.shape
    ret_image = numpy.copy(image)
    max_value = numpy.amax(image)
    for i in range(x_shape):
        for j in range(y_shape):
            if ret_image[i][j] > 0:
                points.append([i,j])
    # only for visualization to plot on top of the original image
    road = imageio.imread('road.png')
    # find four lines
    for i in range(4):
        points, ret_image = find_line(ret_image, points, threshold, inliers, max_value, road)
    imageio.imsave('results/ransac_over_road.png',road)
    return ret_image

def find_line(image,points, threshold, inliers, max_value, road):
    num_of_points = 2
    # keep track of the previous removal set, slope, and intercept
    removal_set = []
    slope = 0
    intercept = 0
    iterations = 0
    horizontal = False
    
    while len(removal_set) < inliers:
        removal_set = []
        horizontal = False
        # generate the two random points
        first_point = numpy.random.randint(len(points))
        second_point = first_point
        while second_point == first_point:
            second_point = numpy.random.randint(len(points))
        first_point = points[first_point]
        second_point = points[second_point]

        # if the slope is not horizontal
        if second_point[0] - first_point[0] != 0:
            slope = (second_point[1]-first_point[1]) / (second_point[0] - first_point[0])
            intercept = int(second_point[1] - slope * second_point[0])
            for point in points:
                if abs(point[1] - (slope * point[0] + intercept)) <= threshold:
                    removal_set.append(point)
        # if horizontal movement
        else:
            horizontal = True
            slope = 0 
            intercept = second_point[0]
            for point in points:
                if abs(point[0] - second_point[0]) <= threshold:
                    removal_set.append(point)
                    
    # remove the points in the target set and highlight the inliers
    for point in removal_set:
        if (point[0] >= 1 and point[0] < image.shape[1]-1 and point[1] >= 1 and point[1] < image.shape[0]-1):
            image[-1+point[0]:point[0]+2, -1+point[1]: point[1] + 2] = max_value
            road[-1+point[0]:point[0]+2, -1+point[1]: point[1] + 2] = 255
        points.remove(point)

    # draw the line
    if not horizontal:
        for i in range(image.shape[0]):
            target = slope * i + intercept
            if target >= 0 and target < image.shape[1]:
                image[i][int(target)] = max_value
                road[i][int(target)] = 255
    else:
        for i in range(image.shape[1]):
            image[intercept][i] = max_value
            road [intercept][i] = 255
    return points,image