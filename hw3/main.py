import imageio
import matplotlib.pyplot as plt
import kmeans
import sys

if __name__ == "__main__":
    image = imageio.imread(sys.argv[1])
    image = image.astype('float32')
    k = int(sys.argv[2])
    kmeansResult = kmeans.kmeans(k,image)
    plt.imshow(kmeansResult)
    plt.show()