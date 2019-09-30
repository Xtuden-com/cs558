import numpy
import clip 

def hough(image, x_buckets, y_buckets):
    H_accumulator = [[0 for _ in range(y_buckets)] for _ in range(x_buckets)]
    x_size, y_size = image.shape
    points = []
    for i in range(x_size-1):
        for j in range(y_size-1):
            if image[i][j] > 0:
                points.append([i,j])
    theta_intervals = 180 / y_buckets
    for point in points:
        for i in range(y_buckets):
            theta = theta_intervals * i
            rho = int(point[0] * numpy.cos(theta) + point[1] * numpy.sin(theta))
            rho += x_buckets// 2
            if rho > 0 and rho < x_buckets: 
                H_accumulator[rho][i] += 100
    return H_accumulator