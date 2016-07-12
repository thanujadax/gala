import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage as ndi


from skimage.morphology import watershed
# from vigra.analysis import watersheds
from skimage.feature import peak_local_max

from PIL import Image

outputFileName = '/home/thanuja/projects/data/toyData/set8/watershed/00.png'
inputFileName = '/home/thanuja/projects/data/toyData/set8/membranes_rfc/00_probability.tif'
image = np.array(Image.open(inputFileName).convert('L'), 'f')
image = (image - 1) * (-1)

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
# (labels,max_region_label) = watersheds(image)


'''
skimage watershed:
from scipy import ndimage
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
markers = morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)
'''
labels_ws = watershed(-distance, markers, mask=image)
from gala import viz
viz.imshow_rand(labels_ws)
# plt.show()
'''
plt.imshow(labels)
plt.show()
'''

# save image as png
scipy.misc.imsave(outputFileName, labels_ws)

