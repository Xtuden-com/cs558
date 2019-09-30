import numpy
import clip 
import matplotlib.pyplot as plt
import math

def hough(image, x_buckets, y_buckets):
    H_accumulator = numpy.array([[0 for _ in range(y_buckets)] for _ in range(x_buckets)])
    x_size, y_size = image.shape
    points = []
    for i in range(x_size-1):
        for j in range(y_size-1): 
            if image[i][j] > 0:
                points.append([i,j])
    theta_intervals = 180 / y_buckets
    arr = []
    x_axis = []
    for point in points:
        for i in range(y_buckets):
            theta = theta_intervals * i
            print(theta)
            rho = int(point[1] * numpy.cos(theta) + point[0] * numpy.sin(theta)) + x_buckets//2
            if rho < x_buckets:
                H_accumulator[rho][i] += 1
    #return H_accumulator
    plt.imshow(H_accumulator,cmap='gray')
    plt.show()
    ret_image = numpy.zeros(image.shape)
    ret_image = extract_line(H_accumulator,ret_image,theta_intervals,x_buckets)
    return extract_line(H_accumulator,ret_image,theta_intervals,x_buckets)

def extract_line(H_accumulator,image, theta_intervals,x_buckets):
    max_value = -1 
    max_row = 0 
    max_col = 0
    x_size, y_size = H_accumulator.shape
    for i in range(x_size):
        for j in range(y_size):
            if H_accumulator[i][j] > max_value:
                max_value = H_accumulator[i][j]
                max_row = i
                max_col = j
    theta = max_col * theta_intervals
    slope = -numpy.cos(theta) / numpy.sin(theta) 
    intercept = math.ceil((max_row-(x_buckets//2)) / numpy.sin(theta))
    H_accumulator[max_row][max_col] = 0
    print(max_row, slope,intercept, theta, max_row)
    draw_line(image,slope,intercept)
    return image


def draw_line(image,slope,intercept):
    x_size, y_size = image.shape
    for i in range(x_size):
        target = slope * i + intercept 
        if target >= 0 and target < y_size: 
            image[i][int(target)] = 255