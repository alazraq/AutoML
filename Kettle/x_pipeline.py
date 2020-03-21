from sklearn.pipeline import Pipeline

from data_imputer import DataImputer
from my_standard_scaler import MyStandardScaler
from data_augmenter import DataAugmenter
from my_one_hot_encoder import MyOneHotEncoder


class XPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
            ('DataImputer', DataImputer()),
            ('MyStandardScaler', MyStandardScaler()),
            ('DataAugmenter', DataAugmenter()),
#             ('MyOneHotEncoder', MyOneHotEncoder()),
        ])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)
