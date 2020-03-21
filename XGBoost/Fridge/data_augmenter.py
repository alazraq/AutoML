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
        x["weekday"] = x.index.dayofweek
        x["month"] = x.index.month
        x["hour"] = x.index.hour
        x["is_weekend"] = (x["weekday"] > 4) * 1
        x["is_holidays"] = (x.index.to_series().apply(lambda t: t in fr_holidays)) * 1
        
        x['weekday_sin'] = np.sin((x.weekday)*(2.*np.pi/7))
        x['weekday_cos'] = np.cos((x.weekday)*(2.*np.pi/7))
        x['month_sin'] = np.sin((x.month-1)*(2.*np.pi/12))
        x['month_cos'] = np.cos((x.month-1)*(2.*np.pi/12))

        x["is_TVtime"] = ((x.hour > 17) & (x.hour < 23)) * 1
        # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
        x["is_night"] = ((x.hour > 0) & (x.hour < 7)) * 1
        
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

#         x['mean_3'] = (x['consumption'].rolling(1).sum().values
#                        + x['consumption'].rolling(1).sum().shift(-1).values) / 3
#         x['mean_5'] = (x['consumption'].rolling(2).sum().values
#                        + x['consumption'].rolling(2).sum().shift(-2).values) / 5
#         x['mean_10'] = (x['consumption'].rolling(5).sum().values
#                         + x['consumption'].rolling(5).sum().shift(-5).values) / 10
#         x['mean_20'] = (x['consumption'].rolling(10).sum().values
#                         + x['consumption'].rolling(10).sum().shift(-10).values) / 20
#         x['mean_30'] = (x['consumption'].rolling(15).sum().values 
#                         + x['consumption'].rolling(15).sum().shift(-15).values) / 30
#         x['mean_41'] = (x['consumption'].rolling(20).sum() + x['consumption'].rolling(20).sum().shift(-20)) / 41
#         x['mean_61'] = (x['consumption'].rolling(30).sum() + x['consumption'].rolling(30).sum().shift(-30)) / 61

        x['rolling_mean_5'] = x['consumption'].rolling(window=5).mean()
        x['rolling_mean_15'] = x['consumption'].rolling(window=15).mean()
        x['rolling_mean_-5'] = x['consumption'].rolling(window=5).mean().shift(-5)
        x['rolling_mean_-15'] = x['consumption'].rolling(window=15).mean().shift(-15)
        
        x['rolling_std_3'] = x['consumption'].rolling(window=3).std()
        x['rolling_std_5'] = x['consumption'].rolling(window=5).std()
        x['rolling_std_-3'] = x['consumption'].rolling(window=3).std().shift(-3)
        x['rolling_std_-5'] = x['consumption'].rolling(window=5).std().shift(-5)
        
        x['rolling_max_3'] = x['consumption'].rolling(window=5).max()
#         x['rolling_max_5'] = x['consumption'].rolling(window=10).max()
        x['rolling_min_3'] = x['consumption'].rolling(window=5).min()
        x = x.ffill().bfill()
        
        return x
