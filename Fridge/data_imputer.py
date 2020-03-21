import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataImputer(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.X = None

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x.index = pd.to_datetime(x.index)
        try:
            x.drop(['Unnamed: 9', 'visibility', 'humidity', 'humidex', 'windchill', 'wind', 'pressure'],
                   axis=1, 
                   inplace=True)
        except KeyError as e:
            pass
        days_to_drop = ["2013-10-27", "2013-10-28", "2013-12-18", "2013-12-19", 
                        "2013-08-01", "2013-08-02", "2013-11-10", "2013-07-07", 
                        "2013-09-07", "2013-03-30", "2013-10-27", "2013-07-14"]
        
        x.drop([pd.to_datetime(day) for day in days_to_drop], inplace=True)
        x = x.interpolate(method='linear').fillna(method='bfill')
        return x
