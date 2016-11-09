import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import h5py
from PIL import Image

# outputH5FileName = '/home/thanuja/projects/data/toyData/set8/watershed_h5/test_ws.h5'
# inputDir = '/home/thanuja/projects/data/toyData/set8/membranes_rfc'

outputH5FileName = '/home/thanuja/DATA/ISBI2012/gala/hdf5/ws-test-rfc.h5'
inputDir = '/home/thanuja/DATA/ISBI2012/gala/flatfiles/test/probMaps_rfc'
# membranes=0

# get all tif files in the input dir
# 8 for test, 1 for train
maxNumIm = 10
binaryThreshold = 0.50
minDistBetwnPeaks = 1
minPeakVal = 0.2 # minimum threshold for peak
sizeR = 512
sizeC = 512
i = 0
for file in sorted(os.listdir(inputDir)):
    if i>maxNumIm :
        break
    # for all the files, perform watershed
    # image = np.array(Image.open(inputFileName).convert('L'), 'f')
    image = np.array(Image.open(os.path.join(inputDir,file)).convert('L'), 'f')
    # normalize to [0,1]
    image = image/(np.max(image))
    # invert image s.t. membrane = 0, cell-interiror = 1
    # image = (image-1) * (-1)
    # produce binary image
    binImage = 1*(image>binaryThreshold)
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(binImage)
    local_maxi = peak_local_max(distance, indices=False, min_distance=minDistBetwnPeaks,threshold_abs=minPeakVal)
    markers = ndi.label(local_maxi)[0]    
    labels = watershed(image, markers, mask=binImage)    

    labels1 = np.empty([1,sizeR,sizeC])
    labels1[0,:,:] = labels
    if i==0 :
        im_array = np.empty([1,sizeR,sizeC])
        im_array[0,:,:] = labels
    else :
        im_array = np.concatenate((im_array,labels1),axis=0)
    i = i+1

with h5py.File(outputH5FileName,'w') as hf:
    hf.create_dataset('stack',data=im_array)


