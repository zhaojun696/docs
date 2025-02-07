#%%
import numpy as np
from numpy import linalg as la
n_components = 3
#缩减的维度
def loadDataSet():
   return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]
dataMat = loadDataSet()
U, Sigma, VT = la.svd(dataMat)
Sigma3 = np.diag(Sigma)
n_size = np.array(Sigma).shape
train_feat_list = U[:,:n_size[0]]*np.mat(Sigma3)*VT[:n_size[0],:n_components]

train_feat_list