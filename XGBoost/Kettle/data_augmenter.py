from sklearn.base import BaseEstimator, TransformerMixin
import holidays
import pandas as pd
import numpy as np


# +
class DataAugmenter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        fr_holidays = holidays.France()
#         x["weekday"] = x.index.dayofweek
        x["month"] = x.index.month
        x["hour"] = x.index.hour
#         x["is_weekend"] = (x["weekday"] > 4) * 1
        x["is_holidays"] = (x.index.to_series().apply(lambda t: t in fr_holidays)) * 1
        
#         x['month_sin'] = np.sin((x.month-1)*(2.*np.pi/12))
#         x['month_cos'] = np.cos((x.month-1)*(2.*np.pi/12))

        x["is_breakfast"] = ((x.hour > 5) & (x.hour < 9)) * 1
        x["is_teatime"] = ((x.hour > 16) & (x.hour < 20)) * 1
        
        x['lag_1'] = x['consumption'].shift(1)
        x['lag_2'] = x['consumption'].shift(2)
        x['lag_3'] = x['consumption'].shift(3)
        x['lag_4'] = x['consumption'].shift(4)
        x['lag_5'] = x['consumption'].shift(5)
        x['lag_10'] = x['consumption'].shift(10)
        x['lag_20'] = x['consumption'].shift(20)
        x['lag_future_1'] = x['consumption'].shift(-1)
        x['lag_future_2'] = x['consumption'].shift(-2)
        x['lag_future_3'] = x['consumption'].shift(-3)
        x['lag_future_4'] = x['consumption'].shift(-4)
        x['lag_future_5'] = x['consumption'].shift(-5)
        x['lag_future_10'] = x['consumption'].shift(-10)
        x['lag_future_20'] = x['consumption'].shift(-20)

        x['rolling_mean_5'] = x['consumption'].rolling(window=3).mean()
        x['rolling_mean_-5'] = x['consumption'].rolling(window=3).mean().shift(-3)

#         x['rolling_std_3'] = x['consumption'].rolling(window=3).std()
#         x['rolling_std_-3'] = x['consumption'].rolling(window=3).std().shift(-3)

#         x['rolling_max_3'] = x['consumption'].rolling(window=3).max()
#         x['rolling_min_3'] = x['consumption'].rolling(window=3).min()
    
        x = x.ffill().bfill()
        
        return x
