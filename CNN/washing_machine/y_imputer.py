import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class YImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        x.index = pd.to_datetime(x.index)
        try:
            x.drop(['fridge_freezer', 'TV', 'kettle'], axis=1, inplace=True)
        except KeyError as e:
            pass
        days_to_drop = ["2013-10-27", "2013-10-28", "2013-12-18", "2013-12-19", 
                "2013-08-01", "2013-08-02", "2013-11-10", "2013-07-07", 
                "2013-09-07", "2013-03-30", "2013-07-14"]
        
        for day in days_to_drop:
            x.drop(x.loc[day].index, inplace=True)
        x = x.interpolate(method='linear').fillna(method='bfill')
#         x.dropna(inplace=True)
        return x
