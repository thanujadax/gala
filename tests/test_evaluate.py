import numpy as np
from numpy.testing import assert_equal
from gala import evaluate as ev

def test_contingency_table():
    seg = np.array([0, 1, 1, 1, 2, 2, 2, 3])
    gt = np.array([1, 1, 1, 2, 2, 2, 2, 0])
    ct = ev.contingency_table(seg, gt, ignore_seg=[], ignore_gt=[])
    ct0 = ev.contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0])
    ctd = ct.todense()
    assert_equal(ctd, np.array([[0.   , 0.125, 0.   ],
                                [0.   , 0.25 , 0.125],
                                [0.   , 0.   , 0.375],
                                [0.125, 0.   , 0.   ]]))
    assert ct.shape == ct0.shape
