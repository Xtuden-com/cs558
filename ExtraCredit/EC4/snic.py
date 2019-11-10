import numpy
import matplotlib.pyplot as plt
import gradient
import convolution
import sys
import heapq

def initialCenters(image):
    # centers are 50 apart starting at 25,25
    colors = []
    centers = []
    xsize, ysize, _ = image.shape
    xblocks = xsize // 50 
    yblocks = ysize // 50
    for i in range(xblocks):
        for j in range(yblocks):
            x = 25 + i * 50
            y = 25 + j * 50
            centers.append([x,y])
            colors.append(image[x][y])
    return centers, colors

def drawBorders(labelMap, ret):
    xsize, ysize = labelMap.shape
    for i in range(xsize-1):
        for j in range(ysize-1):
            if labelMap[i][j] != labelMap[i+1][j] or labelMap[i][j] != labelMap[i][j+1]:
                ret[i][j] = [0,0,0]
    return ret

def updateCluster(clusters, index, element):
    cluster = clusters[index]
    # increment count
    elements = cluster[5]
    cluster[5] += 1
    # the index must be an integer
    cluster[0] = ((cluster[0] * elements) + element[0]) // cluster[5]
    cluster[1] = ((cluster[1] * elements) + element[1])  // cluster[5]
    for i in range(2,5):
        cluster[i] = ((cluster[i] * elements) + element[i]) / cluster[5]

def squaredDifference(a,b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total

def elementDistance(a,b,s,m):
    pixelDistance = squaredDifference([a[0],a[1]],[b[0],b[1]]) / s
    colorDistance = squaredDifference(a[2:],b[2:]) / m 
    return (pixelDistance + colorDistance) ** (1/2)

def snic(image, compactnessFactor):
    m = compactnessFactor
    heap = []
    centers, colors = initialCenters(image)
    s = (image.shape[0] * image.shape[1] / len(centers)) ** (1/2)
    xsize, ysize, _ = image.shape    
    ret = numpy.zeros(image.shape)
    labelMap = numpy.zeros((xsize,ysize))
    # elements in cluster are x, y, R, G, B, count
    clusters = [[0,0,0,0,0,0] for i in range(len(centers))]
    directions =[ [-1, 0], [-1,-1] , [-1,1] , [1,-1], [1, 0] , [1,1], [0,1], [0,-1] ]
    for i in range(len(centers)):
        center = centers[i]
        color = colors[i]
        # order is distance, k, R, G, B 
        # sort by distance
        element = (0,i+1,color[0],color[1],color[2],center[0],center[1])
        heapq.heappush(heap,element)
    while len(heap) != 0:
        target = heapq.heappop(heap)
        distance, k, colorR, colorG, colorB, x, y = target
        if labelMap[x][y] == 0: 
            labelMap[x][y] = k           
            updateCluster(clusters, k-1, [x,y,colorR,colorG,colorB])
            for move in directions:
                moveX = x + move[0]
                moveY = y + move[1]
                if moveX >= 0 and moveX < xsize and moveY >= 0 and moveY < ysize:
                    if labelMap[moveX][moveY] == 0:
                        old = [x,y, colorR,colorG, colorB]
                        targetColor = image[moveX,moveY]
                        target = [moveX, moveY, targetColor[0], targetColor[1],targetColor[2]]
                        d = elementDistance(target,old,s,m)
                        e = (d, k, targetColor[0], targetColor[1], targetColor[2], moveX, moveY)
                        heapq.heappush(heap,e)
    for i in range(xsize):
        for j in range(ysize):
            clusterIndex = labelMap[i][j]
            target = clusters[int(clusterIndex-1)]
            ret[i][j] = [target[2]/255,target[3]/255,target[4]/255]
    plt.imshow(ret)
    plt.show()
    return drawBorders(labelMap,ret)