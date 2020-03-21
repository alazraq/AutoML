import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class YImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        try:
            x.drop(['kettle', 'fridge_freezer', 'washing_machine'], axis=1, inplace=True)
        except KeyError as e:
            pass
        x = x.interpolate(method='linear').fillna(method='bfill')
        x.index = pd.to_datetime(x.index)
        return x
