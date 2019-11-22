import numpy 
import imageio
import sys 
import glob
import matplotlib.pyplot as plt
import heapq

def generateHistogram(image,buckets):
    histogram = [[] for _ in range(buckets)]
    x_size, y_size = image.shape
    step_size = 255 / buckets
    for i in range(x_size):
        for j in range(y_size):
            pixel = image[i,j]
            index = 0
            if pixel == 255:
                index = buckets - 1
            else:
                index = pixel//step_size
            histogram[int(index)].append([i,j])
    return histogram

def verify(histograms, x_size, y_size):
    image = numpy.zeros((x_size,y_size))
    for histogram in histograms: 
        for buckets in histogram:
            for pixel in buckets:
                image[pixel[0],pixel[1]] += 1
    for i in range(x_size):
        for j in range(y_size):
            if image[i][j] != 3:
                return False
    return True

def generateFeatures(histograms):
    ret = []
    for histogram in histograms:
        for bucket in histogram:
            ret.append(len(bucket))
    return ret

def trainData(dataTitle, trainingSets, buckets):
    ret = []
    for i in range(len(dataTitle)):
        currentClass = dataTitle[i]
        trainingSet = trainingSets[i]
        for path in trainingSet:
            image = imageio.imread(path)
            x_size, y_size, _ = image.shape
            histogramR = generateHistogram(image[:,:,0],buckets)
            histogramG = generateHistogram(image[:,:,1],buckets)
            histogramB = generateHistogram(image[:,:,2],buckets)
            assert verify([histogramR,histogramG,histogramB], x_size, y_size) == True
            features = generateFeatures([histogramR,histogramG,histogramB])
            ret.append((features,currentClass))
    return ret


def euclideanDistance(vector1,vector2):
    ret = 0
    for i in range(len(vector1)):
        ret += (vector1[i] - vector2[i]) ** 2
    return ret ** (1/2) 

def findClass(image,trainingData, buckets):
    histogramR = generateHistogram(image[:,:,0],buckets)
    histogramG = generateHistogram(image[:,:,1],buckets)
    histogramB = generateHistogram(image[:,:,2],buckets)
    features = generateFeatures([histogramR,histogramG,histogramB])
    maxHeap = []
    for i in range(len(trainingData)):
        targetFeatures, targetClass = trainingData[i]
        distance = euclideanDistance(targetFeatures,features)
        heapq.heappush(maxHeap,(distance,targetClass))
    _ , retClass = heapq.heappop(maxHeap)
    return retClass


def classifyImages(buckets):
    root_dir = 'data/ImClass/'
    dataTitle = ['coast', 'forest', 'insidecity']
    trainingSets = []
    testingSets = []
    for i in range(3):
        trainingSets.append(glob.glob(root_dir+dataTitle[i]+'_train?.jpg'))
        testingSets.append(glob.glob(root_dir+dataTitle[i]+'_test?.jpg'))
        testingSets[i].sort()
    trainingData = trainData(dataTitle,trainingSets,buckets)
    accuracy = 0
    for i in range(len(testingSets)):
        targetClass = testingSets[i]
        for j in range(len(targetClass)):
            targetImage = imageio.imread(targetClass[j])
            foundClass = findClass(targetImage,trainingData,buckets)
            if foundClass == dataTitle[i]:
                accuracy += 1
            print ('Test image ',j+1,' of class ',dataTitle[i],' has been assigned to class ',foundClass)
    return accuracy / 12

if __name__ == "__main__":
    print('Accuracy is: ', classifyImages(10))