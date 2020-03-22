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

        x["is_breakfast"] = ((x.hour > 5) & (x.hour < 9)) * 1
        x["is_teatime"] = ((x.hour > 16) & (x.hour < 20)) * 1
        x["is_TVtime"] = ((x.hour > 17) & (x.hour < 23)) * 1
        # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
        x["is_night"] = ((x.hour > 0) & (x.hour < 7)) * 1
        return x
