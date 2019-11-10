import matplotlib.pyplot as plt 
import imageio 
import sys
import snic

if __name__ == "__main__":
    image = imageio.imread(sys.argv[1]).astype('float32')
    compression = int(sys.argv[2])
    plt.imshow(snic.snic(image,compression))
    plt.show()