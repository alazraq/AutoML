from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class RNNDataFormatter(BaseEstimator, TransformerMixin):
    
    def __init__(self, batch_size=5):
        self.X = None
        self.batch_size = batch_size
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        
        xx = np.zeros((x.shape[0], self.batch_size, 1))
        x = np.pad(x, ((self.batch_size//2, self.batch_size//2), (0,0)), 'mean')
        for i in range(len(xx)):
        #     print(x[i:60+i, :].shape)
            try:
                xx[i, :, :] = x[i:self.batch_size+i, :]
            except:
                print(i)
        return xx
