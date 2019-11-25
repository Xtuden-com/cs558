import numpy 
import matplotlib.pyplot as plt 
from imageClassification import generateHistogram, generateFeatures
import glob
import imageio

def meanShift(descriptors):
    ret = []
    converge = False
    

if __name__ == "__main__":
    data1 = glob.glob('data1/*.jpg')
    descriptors = []
    # iterate through images
    for path in data1:
        image = imageio.imread(path)
        