from sklearn.pipeline import Pipeline

from data_imputer import DataImputer
from my_standard_scaler import MyStandardScaler
from data_augmenter import DataAugmenter
from my_one_hot_encoder import MyOneHotEncoder
from rnn_data_formatter import RNNDataFormatter



class XPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
    			('DataImputer', DataImputer()),
    			('MyStandardScaler', MyStandardScaler()),
    			('DataAugmenter', DataAugmenter()),
    			('MyOneHotEncoder', MyOneHotEncoder()),
  			('RNNDataFormatter', RNNDataFormatter())
	])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)
