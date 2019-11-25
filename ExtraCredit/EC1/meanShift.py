import numpy 
import matplotlib.pyplot as plt 
import glob
import imageio
from skimage import exposure,img_as_float
import copy

# assign descriptors to the index of center they converged to
def indexDescriptors(means):
    means = means.tolist()
    swap = []
    for temp in means:
        swap.append(tuple(temp))
    means = copy.deepcopy(swap)
    uniqueValues = list(set(swap))
    ret = []
    # iterate through each descriptor
    for i in range(len(means)):
        target = means[i]
        # if the descriptor equals the center assign the index value
        for j in range(len(uniqueValues)):
            center = uniqueValues[j]
            if target == center:
                ret.append(j)
                break
    return ret

def meanShift(descriptors, maxDistance):
    means = numpy.copy(descriptors)
    for i in range(len(means)):
        target = means[i]
        surroundings = []
        converged = False
        while not converged:
            for temp in descriptors:
                # "kernel" function
                # if the distance to the next descriptor is less than the maxDistance you can include it
                # with the calculation of the mean with weight 1
                distance = numpy.linalg.norm(target-temp)
                if distance <= maxDistance:
                    surroundings.append(temp)
            center = numpy.mean(surroundings,axis=0)
            if numpy.linalg.norm(target-center) == 0:
                converged = True
            target = center
            surroundings = []
        means[i] = target 
    ret = indexDescriptors(means)
    return ret

# concatenate each histogram to a 24 dimension list
def generateFeatures(histograms):
    ret = []
    for histogram in histograms:
        ret.extend(histogram)
    return ret

def cluster(path,maxDistance):
    data = glob.glob(path)
    descriptors = []
    images = []
    # iterate through images
    for path in data:
        image = img_as_float(imageio.imread(path))
        histogramR,_ = exposure.histogram(image[:,:,0],nbins=8)
        histogramG,_ = exposure.histogram(image[:,:,1],nbins=8)
        histogramB,_ = exposure.histogram(image[:,:,2],nbins=8)
        descriptor = generateFeatures([histogramR,histogramG,histogramB])
        descriptors.append(descriptor)
        images.append((images,path))
    # 1880000 for 3 cluster
    # 1800000 for 5 clusters but good seperation
    indexValues = meanShift(descriptors,maxDistance)  
    numIndex = len(set(indexValues)) 
    print(numIndex, ' Clusters')
    imageSets = [[] for _ in range(numIndex)]
    for i in range(len(images)):
        targetIndex = indexValues[i]
        imageSets[targetIndex].append(images[i])
    for i in range(numIndex):
        print('Cluster ', i, ': ', end='')
        for j in range(len(imageSets[i])):
            print(imageSets[i][j][1],', ', end='')
        print('')

if __name__ == "__main__":
    print('Data Set 1:')
    cluster("data1/*.jpg",1880000)
    print('Data Set 2:')
    cluster("data2/*.jpg",1920000)