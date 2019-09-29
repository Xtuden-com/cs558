import convolution
import imageio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = imageio.imread('road.png')
    image = convolution.convulveGaussian(image,1)
    plt.imshow(image, cmap=plt.get_cmap(name="gray"))
    plt.show()