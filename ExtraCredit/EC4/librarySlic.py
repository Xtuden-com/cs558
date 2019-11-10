from skimage import data, io, segmentation, color
import imageio 
import matplotlib.pyplot as plt

img = imageio.imread('wt_slic.png')
xsize, ysize, _ = img.shape
s = xsize // 50 * ysize // 50
segments = segmentation.slic(img, n_segments=s, compactness=10)
out1 = color.label2rgb(segments, img, kind='avg')
plt.imshow(out1)
plt.show()
imageio.imwrite('results/librayslic.png',out1)
processed = segmentation.mark_boundaries(out1, segments)
plt.imshow(processed)
plt.show()
imageio.imwrite('results/librayslic_borders.png',processed)