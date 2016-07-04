import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage as ndi

# from skimage.morphology import watershed
from vigra.analysis import watersheds
from skimage.feature import peak_local_max

from PIL import Image

outputFileName = '/home/thanuja/projects/data/toyData/set8/watershed/train_ws.h5'
inputDir = '/home/thanuja/projects/data/toyData/set8/membranes_rfc'

# get all tif files in the input dir
i = 0
for file in sorted(os.listdir(inputDir)):
    # for all the files, perform watershed
    image = np.array(Image.open(inputFileName).convert('L'), 'f')
    image = (image - 1) * (-1)
    (labels,max_region_label) = watersheds(image)
    if i=0 :
        im_array = labels
    else :
        im_array = np.concatenate(im_array,labels,axis=0)
    i = i+1

with h5py.File(outputH5fileName,'w') as hf:
    hf.create_dataset('stack',data=im_array)


