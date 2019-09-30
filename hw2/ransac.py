import numpy
import sys
import matplotlib.pyplot as plt

def ransac(image, threshold, inliers):
    points = []
    x_shape, y_shape = image.shape
    ret_image = numpy.copy(image)
    # remove the border since we need to place a
    # 3x3 block around them
    for i in range(1,x_shape-1):
        for j in range(1,y_shape-1):
            if ret_image[i][j] > 0:
                points.append([i,j])
    # find four lines
    for i in range(4):
        points, ret_image = find_line(ret_image, points, threshold, inliers)
    return ret_image

def find_line(image,points, threshold, inliers):
    num_of_points = 2
    """
    error = .5
    sample_count = 0
    N = sys.maxsize
    p = .95
    """
    # keep track of the previous removal set, slope, and intercept
    removal_set = []
    slope = 0
    intercept = 0
    iterations = 0
    
    while len(removal_set) < inliers:
        removal_set = []
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
        # if vertical only compare x movement
        else:
            slope = 0 
            intercept = second_point[0]
            for point in points:
                if abs(point[0] - second_point[0]) <= threshold:
                    removal_set.append(point)
        """
        # lower error, N, and increase step count
        e = 1 - len(removal_set) / len(points)
        if e != 1:
            N = numpy.log(1-p) / numpy.log(1-(1-e)**num_of_points)
        sample_count+=1
        """
        iterations+=1

    print(len(removal_set), len(points), slope, intercept,iterations)
    # remove the points in the target set and highlight the inliers
    for point in removal_set:
        image[-1+point[0]:point[0]+2, -1+point[1]: point[1] + 2] = 255
        points.remove(point)

    # draw the line
    if slope != 0:
        for i in range(image.shape[0]):
            target = slope * i + intercept
            if target >= 0 and target < image.shape[1]:
                image[i][int(target)] = 255
    else:
        for i in range(image.shape[1]):
            image[intercept][i] = 255
    return points,image