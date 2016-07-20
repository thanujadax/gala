import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage as ndi

from skimage.morphology import watershed
# from vigra.analysis import watersheds
from skimage.feature import peak_local_max
import h5py
from PIL import Image

outputH5FileName = '/home/thanuja/projects/data/toyData/set8/watershed_h5/train_ws.h5'
inputDir = '/home/thanuja/projects/data/toyData/set8/membranes_rfc'

# get all tif files in the input dir
# 8 for test, 1 for train
maxNumIm = 1
binaryThreshold = 5
minDistBetwnPeaks = 1
i = 0
for file in sorted(os.listdir(inputDir)):
    if i>maxNumIm :
        break
    # for all the files, perform watershed
    # image = np.array(Image.open(inputFileName).convert('L'), 'f')
    image = np.array(Image.open(os.path.join(inputDir,file)).convert('L'), 'f')
    binImage = 1*(image>binaryThreshold)
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(binImage)
    local_maxi = peak_local_max(distance, indices=False, min_distance=minDistBetwnPeaks)
    markers = ndi.label(local_maxi)[0]    
    labels = watershed(image, markers, mask=image)    

    labels1 = np.empty([1,500,500])
    labels1[0,:,:] = labels
    if i==0 :
        im_array = np.empty([1,500,500])
        im_array[0,:,:] = labels
    else :
        im_array = np.concatenate((im_array,labels1),axis=0)
    i = i+1

with h5py.File(outputH5FileName,'w') as hf:
    hf.create_dataset('stack',data=im_array)


