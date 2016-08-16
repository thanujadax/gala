# imports
from gala import imio, classify, features, agglo, evaluate as ev
import numpy as np
#import scipy.misc.imsave
import os
from PIL import Image

'''
Inputs
'''
'''
h5File_train_gt = '/home/thanuja/projects/external/gala/tests/example-data/train-gt.lzf.h5'
h5File_train_ws = '/home/thanuja/projects/external/gala/tests/example-data/train-ws.lzf.h5'
h5File_train_probMap = '/home/thanuja/projects/external/gala/tests/example-data/train-p1.lzf.h5'

h5File_test_ws = '/home/thanuja/projects/external/gala/tests/example-data/test-ws.lzf.h5'
h5File_test_probMap = '/home/thanuja/projects/external/gala/tests/example-data/test-p1.lzf.h5'
'''


h5File_train_gt = '/home/thanuja/projects/data/drosophilaLarva_ssTEM/dataset01_hdf5/train/train_gt.h5'
h5File_train_ws = '/home/thanuja/projects/data/drosophilaLarva_ssTEM/dataset01_hdf5/train/train_ws.h5'
h5File_train_probMap = '/home/thanuja/projects/data/drosophilaLarva_ssTEM/dataset01_hdf5/train/train_probMaps.h5'

h5File_test_ws = '/home/thanuja/projects/data/drosophilaLarva_ssTEM/dataset01_hdf5/test/test_ws.h5'
h5File_test_probMap = '/home/thanuja/projects/data/drosophilaLarva_ssTEM/dataset01_hdf5/test/test_probMaps.h5'


'''
Outputs
'''
outputRoot = '/home/thanuja/projects/RESULTS/gala/20160816'

# read in training data
# groundtruth volume, probability maps, superpixe/watershed map
gt_train, pr_train, ws_train = (map(imio.read_h5_stack,
                                [h5File_train_gt, h5File_train_probMap,
                                 h5File_train_ws]))

# create a feature manager
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])

# create Region Adjacency Graph (RAG) and obtain a training dataset
g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
(X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc)[0]
y = y[:, 0] # gala has 3 truth labeling schemes, pick the first one ????
print((X.shape, y.shape)) # standard scikit-learn input format

# train a classifier, scikit-learn syntax
rf = classify.DefaultRandomForest().fit(X, y)
# a policy is the composition of a feature map and a classifier
# policy = merge priority function
learned_policy = agglo.classifier_probability(fc, rf)

# get the test data and make a RAG with the trained policy
pr_test, ws_test = (map(imio.read_h5_stack,
                        [h5File_test_probMap, h5File_test_ws]))
g_test = agglo.Rag(ws_test, pr_test, learned_policy, feature_manager=fc)
g_test.agglomerate(0.5) # best expected segmentation obtained with a threshold of 0.5
seg_test1 = g_test.get_segmentation()

# convert hdf into png and save 
np_data = np.array(seg_test1)
sizeZ,sizeY,sizeX = np_data.shape
for i in range(0,sizeZ):
    im1 = np_data[i,:,:]
    im = Image.fromarray(im1.astype('uint8'))
    imFileName = str(i).zfill(3) + ".png"
    imFileName = os.path.join(outputRoot,imFileName)
    #scipy.misc.toimage(im, cmin=0.0, cmax=...).save(imFileName)
    im.save(imFileName)


'''
###############################################################
# the same approach works with a multi-channel probability map
p4_train = imio.read_h5_stack('train-p4.lzf.h5')
# note: the feature manager works transparently with multiple channels!
g_train4 = agglo.Rag(ws_train, p4_train, feature_manager=fc)
(X4, y4, w4, merges4) = g_train4.learn_agglomerate(gt_train, fc)[0]
y4 = y4[:, 0]
print((X4.shape, y4.shape))

rf4 = classify.DefaultRandomForest().fit(X4, y4)
learned_policy4 = agglo.classifier_probability(fc, rf4)
p4_test = imio.read_h5_stack('test-p4.lzf.h5')
g_test4 = agglo.Rag(ws_test, p4_test, learned_policy4, feature_manager=fc)
g_test4.agglomerate(0.5)
# extract the segmentation from the RAG model
seg_test4 = g_test4.get_segmentation()

# gala allows implementation of other agglomerative algorithms, including
# the default, mean agglomeration
g_testm = agglo.Rag(ws_test, pr_test,
                    merge_priority_function=agglo.boundary_mean)
g_testm.agglomerate(0.5)
seg_testm = g_testm.get_segmentation()

# examine how well we did with either learning approach, or mean agglomeration
gt_test = imio.read_h5_stack('test-gt.lzf.h5')
import numpy as np
results = np.vstack((
    ev.split_vi(ws_test, gt_test),
    ev.split_vi(seg_testm, gt_test),
    ev.split_vi(seg_test1, gt_test),
    ev.split_vi(seg_test4, gt_test)
    ))

print(results)
'''
