import numpy 
import imageio
import sys 
import glob
import matplotlib.pyplot as plt
import heapq

def generateHistogram(image,buckets):
    histogram = [[] for _ in range(buckets)]
    x_size, y_size = image.shape
    # the pixel values that buckets are spread apart
    step_size = 255 / buckets
    for i in range(x_size):
        for j in range(y_size):
            pixel = image[i,j]
            index = 0
            # if you are at the end add it to the last bucket
            if pixel == 255:
                index = buckets - 1
            else:
                index = pixel//step_size
            # add the pixel location to each bucket
            histogram[int(index)].append([i,j])
    return histogram

# increment count on each pixel as you find it in the histograms
# ensure that each image has 3 count
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

# concatenate each histogram to a 24 dimension list
def generateFeatures(histograms):
    ret = []
    for histogram in histograms:
        for bucket in histogram:
            ret.append(len(bucket))
    return ret

def trainData(dataTitle, trainingSets, buckets):
    ret = []
    for i in range(len(dataTitle)):
        # all the training images corresponding to the current class
        currentClass = dataTitle[i]
        trainingSet = trainingSets[i]
        for path in trainingSet:
            image = imageio.imread(path)
            x_size, y_size, _ = image.shape
            # generate each histogram
            histogramR = generateHistogram(image[:,:,0],buckets)
            histogramG = generateHistogram(image[:,:,1],buckets)
            histogramB = generateHistogram(image[:,:,2],buckets)
            # verify that pixels have been counted 3 times
            assert verify([histogramR,histogramG,histogramB], x_size, y_size) == True
            features = generateFeatures([histogramR,histogramG,histogramB])
            # return the features and class as a tuple
            ret.append((features,currentClass))
    return ret

# basic l2 norm
def euclideanDistance(vector1,vector2):
    ret = 0
    for i in range(len(vector1)):
        ret += (vector1[i] - vector2[i]) ** 2
    return ret ** (1/2) 

def findClass(image,trainingData, buckets):
    # generate a histogram for each color channel
    histogramR = generateHistogram(image[:,:,0],buckets)
    histogramG = generateHistogram(image[:,:,1],buckets)
    histogramB = generateHistogram(image[:,:,2],buckets)
    features = generateFeatures([histogramR,histogramG,histogramB])
    minHeap = []
    # loop through training data and use a min heap to maintain the closest 
    # training image described through the 24 element descriptor
    for i in range(len(trainingData)):
        targetFeatures, targetClass = trainingData[i]
        distance = euclideanDistance(targetFeatures,features)
        heapq.heappush(minHeap,(distance,targetClass))
    _ , retClass = heapq.heappop(minHeap)
    return retClass


def classifyImages(buckets):
    root_dir = 'data/ImClass/'
    # the classes as specified by the file
    dataTitle = ['coast', 'forest', 'insidecity']
    trainingSets = []
    testingSets = []
    # get all the file path names according to class
    for i in range(3):
        trainingSets.append(glob.glob(root_dir+dataTitle[i]+'_train?.jpg'))
        testingSets.append(glob.glob(root_dir+dataTitle[i]+'_test?.jpg'))
        # sort the testingSet so its image1 to image4 for each class
        testingSets[i].sort()
    # generate the feature descriptors of all the trainingData
    trainingData = trainData(dataTitle,trainingSets,buckets)
    accuracy = 0
    for i in range(len(testingSets)):
        targetClass = testingSets[i]
        for j in range(len(targetClass)):
            targetImage = imageio.imread(targetClass[j])
            # the class of the test image's neighbor
            foundClass = findClass(targetImage,trainingData,buckets)
            if foundClass == dataTitle[i]:
                accuracy += 1
            print ('Test image ',j+1,' of class ',dataTitle[i],' has been assigned to class ',foundClass)
    # hardcoded 12 as the the number of testing images
    return accuracy / 12