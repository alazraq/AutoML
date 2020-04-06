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
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        print(x.shape)
        print(x.__class__.__name__)
        while x.shape[0] % self.batch_size != 0:
            print("Appending a row")
            print([x[-1, :]])
            x = np.append(x, [x[-1, :]], axis=0)
        print(x.shape)
        nb_col = x.shape[1]
        return x.reshape((int(x.shape[0] / self.batch_size), self.batch_size, nb_col))
