from sklearn.base import BaseEstimator, TransformerMixin
import holidays
import pandas as pd


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

        x["is_TVtime"] = ((x.hour > 17) & (x.hour < 23)) * 1
        # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
        x["is_night"] = ((x.hour > 0) & (x.hour < 7)) * 1
        
        x['lag_1'] = x['consumption'].shift(1)
        x['lag_5'] = x['consumption'].shift(5)
        x['lag_10'] = x['consumption'].shift(10)
        x['lag_20'] = x['consumption'].shift(20)
        x['lag_25'] = x['consumption'].shift(25)
        x['lag_30'] = x['consumption'].shift(30)
        x['lag_35'] = x['consumption'].shift(35)
        x['lag_40'] = x['consumption'].shift(40)

        x['lag_future_1'] = x['consumption'].shift(-1)
        x['lag_future_5'] = x['consumption'].shift(-5)
        x['lag_future_10'] = x['consumption'].shift(-10)
        x['lag_future_20'] = x['consumption'].shift(-20)
        x['lag_future_25'] = x['consumption'].shift(-25)
        x['lag_future_30'] = x['consumption'].shift(-30)
        x['lag_future_35'] = x['consumption'].shift(-35)
        x['lag_future_40'] = x['consumption'].shift(-40)

        x['rolling_mean_5'] = x['consumption'].rolling(window=5).mean()
        x['rolling_mean_15'] = x['consumption'].rolling(window=15).mean()
        x['rolling_mean_-5'] = x['consumption'].rolling(window=5).mean().shift(-5)
        x['rolling_mean_-15'] = x['consumption'].rolling(window=15).mean().shift(-15)

        x['rolling_std_5'] = x['consumption'].rolling(window=5).std()
        x['rolling_std_15'] = x['consumption'].rolling(window=15).std()
        x['rolling_std_-5'] = x['consumption'].rolling(window=5).std().shift(-5)
        x['rolling_std_-55'] = x['consumption'].rolling(window=15).std().shift(-15)     

        x['rolling_max_10'] = x['consumption'].rolling(window=10).max()
#         x['rolling_max_5'] = x['consumption'].rolling(window=10).max()
        x['rolling_min_10'] = x['consumption'].rolling(window=10).min()
        x = x.ffill().bfill()
        
        return x
