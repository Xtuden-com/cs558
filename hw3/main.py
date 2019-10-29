import imageio
import matplotlib.pyplot as plt
import sys

import kmeans
import slic

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Useage: python main.py kmeansImage kCenters slicImage')
        exit(1)
    imageKmeans = imageio.imread(sys.argv[1]).astype('float32')
    k = int(sys.argv[2])
    imageSlic = imageio.imread(sys.argv[3]).astype('float32')
    kmeansResult = kmeans.kmeans(k,imageKmeans)
    plt.imshow(kmeansResult)
    plt.show()
    slicResult = slic.slic(imageSlic)
    plt.imshow(slicResult)
    plt.show()