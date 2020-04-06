from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rnn_data_formatter import RNNDataFormatter
from y_imputer import YImputer


class YPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
            ('YImputer', YImputer()),
        #   ('StandardScaler', StandardScaler()),
            ('RNNDataFormatter', RNNDataFormatter())
        ])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)

