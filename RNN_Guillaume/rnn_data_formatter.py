from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class RNNDataFormatter(BaseEstimator, TransformerMixin):
    
    def __init__(self, batch_size=60):
        self.X = None
        self.batch_size = batch_size
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        nb_col = x.shape[1]
        print(x.shape)
        xx = np.zeros((x.shape[0], self.batch_size, nb_col))
        x = np.pad(x, ((self.batch_size//2, self.batch_size//2), (0,0)), 'mean')
        print(xx.shape)
        print(x.shape)
        for i in range(len(xx)):
        #     print(x[i:60+i, :].shape)
            try:
                xx[i, :, :] = x[i:self.batch_size+i, :]
            except:
                raise ValueError("g")
                print(i)
        return xx
