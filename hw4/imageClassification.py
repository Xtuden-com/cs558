import numpy 
import imageio
import sys 
import glob


def trainData(dataTitle, trainingSets):
    print('None')

def classifyImages(buckets):
    root_dir = 'data/ImClass/'
    dataTitle = ['coast', 'forest', 'insidecity']
    trainingSets = []
    testingSets = []
    for i in range(3):
        trainingSets.append(glob.glob(root_dir+dataTitle[i]+'_train?.jpg'))
        testingSets.append(glob.glob(root_dir+dataTitle[i]+'_test?.jpg'))
        trainingSets[i].sort()
        testingSets[i].sort()

    trainingData = trainData(dataTitle,trainingSets)
    

if __name__ == "__main__":
    classifyImages(8)