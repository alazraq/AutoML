from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_augmenter import DataAugmenter
from data_imputer import DataImputer
from rnn_data_formatter import RNNDataFormatter


class XPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
            ('DataImputer', DataImputer()),
            ('DataAugmenter', DataAugmenter()),
        #   ('StandardScaler', StandardScaler()),
            ('RNNDataFormatter', RNNDataFormatter())
        ])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)
