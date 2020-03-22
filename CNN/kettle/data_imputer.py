import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataImputer(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.X = None

    def fit(self, x, y=None):
        self.days_to_drop = ["2013-10-27", "2013-10-28", "2013-12-18", "2013-12-19",
                             "2013-08-01", "2013-08-02", "2013-11-10", "2013-07-07",
                             "2013-09-07", "2013-03-30", "2013-07-14"]
        return self

    def transform(self, x, y=None):
        x.index = pd.to_datetime(x.index)     
        try:
            x.drop(['visibility', 'temperature', 'humidity', 'humidex',
                    'windchill', 'wind', 'pressure', 'Unnamed: 9'], axis=1, inplace=True)
            for day in self.days_to_drop:
                x.drop(x.loc[day].index, inplace=True)
        except KeyError as e:
            pass

        x = x.interpolate(method='linear').fillna(method='bfill')
        return x
