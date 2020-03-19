import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class MyStandardScaler(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.columns = X.columns
        self.index = X.index
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(self.scaler.transform(X),
                         columns=self.columns, 
                         index=self.index
                        )
        return X
