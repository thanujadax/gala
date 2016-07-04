import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage as ndi


# from skimage.morphology import watershed
from vigra.analysis import watersheds
from skimage.feature import peak_local_max

from PIL import Image

outputFileName = '/home/thanuja/projects/data/toyData/set8/watershed/09.png'
inputFileName = '/home/thanuja/projects/data/toyData/set8/membranes_rfc/09_probability.tif'
image = np.array(Image.open(inputFileName).convert('L'), 'f')
image = (image - 1) * (-1)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndi.label(local_maxi)[0]
# skimage watershed
# labels = watershed(-distance, markers, mask=image)

''' vigra watershed
watersheds(image, neighborhood=4, seeds = None, methods = ‘RegionGrowing’,
terminate=CompleteGrow, threshold=0, out = None) -> (labelimage, max_ragion_label)
plt.imshow(labels)
'''
'''
wsInput = np.float32(distance)
wsInput.dType
(labels,max_region_label) = watersheds(wsInput, seeds = markers, method = 'RegionGrowing', terminate='CompleteGrow')
'''
(labels,max_region_label) = watersheds(image)

'''
fig, axes = plt.subplots(ncols=3, figsize=(8, 2.7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax0, ax1, ax2 = axes

ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax0.set_title('Overlapping objects')
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title('Distances')
ax2.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
ax2.set_title('Separated objects')

for ax in axes:
    ax.axis('off')

fig.subplots_adjust(hspace=0.01, wspace=0.01, top=0.9, bottom=0, left=0,
                    right=1)
'''
'''
plt.imshow(labels)
plt.show()
'''

# save image as png
scipy.misc.imsave(outputFileName, labels)

