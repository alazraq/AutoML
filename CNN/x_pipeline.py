from sklearn.pipeline import Pipeline

from data_imputer import DataImputer
from rnn_data_formatter import RNNDataFormatter
from sklearn.preprocessing import StandardScaler


class XPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
                ('DataImputer', DataImputer()),
                ('StandardScaler', StandardScaler()),
                ('RNNDataFormatter', RNNDataFormatter())
        ])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)
