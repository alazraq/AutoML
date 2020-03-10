---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Trying to fit GBM

```python
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import holidays
fr_holidays = holidays.France()
import math as mt

import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, SimpleRNN
from keras.models import Sequential

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import datetime
```

```python
X_train = pd.read_csv(
    'provided_data_and_metric/X_train_6GWGSxz.csv',
)
Y_train = pd.read_csv(
    'provided_data_and_metric/y_train_2G60rOL.csv',
)
X_test = pd.read_csv(
    'provided_data_and_metric/X_test_c2uBt2s.csv', 
)
```

```python
class DataImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        try:
            X.drop('Unnamed: 9', axis = 1, inplace = True)
        except KeyError as e:
            pass
        X = X.interpolate(method='linear').fillna(method='bfill')
        return X
class YImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X = X.interpolate(method='linear').fillna(method='bfill')
        return X
class DataAugmenter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X["time_step"] = pd.to_datetime(X["time_step"])
        X["weekday"] = X.time_step.dt.dayofweek
        X["week"] = X.time_step.dt.week
        X["month"] = X.time_step.dt.month
        X["hour"] = X.time_step.dt.hour
        X["minute"] = X.time_step.dt.minute
        X["is_weekend"] = np.zeros(X.shape[0])  
        X.loc[X["weekday"] > 4, "is_weekend"] = 1
        X["is_holidays"] = np.zeros(X.shape[0])  
        X.loc[X.time_step.dt.date.isin(fr_holidays), "is_holidays"] = 1
        X.drop(["time_step", "visibility", "humidity", "humidex", "windchill", "wind", "pressure"], axis=1, inplace=True)
        
        return X
```

```python
p1 = Pipeline([
    (
        '1',
        DataImputer()
    ),
    (
        '2',
        DataAugmenter()
    )
])
p2 = Pipeline([
    (
        '1',
        YImputer()
    )
])
```

```python
X = p1.transform(X_train)
X_t = p1.transform(X_test)
X.shape
```

```python
Y = p2.transform(Y_train)
Y.head()
```

```python
#X["TV"] = Y["TV"]
#X["washing_machine"] = Y["washing_machine"]
#X["fridge_freezer"] = Y["fridge_freezer"]
#X["kettle"] = Y["kettle"]
```

```python
y1 = Y["TV"]
y2 = Y["kettle"]
y3 = Y["washing_machine"]
y4 = Y["fridge_freezer"]
```

```python
X.head()
```

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import ensemble
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler

class Regressor(BaseEstimator):
    def __init__(self):
        self.scaler = StandardScaler()
        params = {'learning_rate': 0.1,
            'max_depth': 8,
            'max_features': 10,
            'min_samples_leaf': 13,
            'n_estimators': 3000,
            'min_samples_split': 14}

        params_2 = {'learning_rate': 0.1,
            'max_depth': 10,
            'max_features': 0.4,
            'min_samples_leaf': 9,
            'min_samples_split': 10}
        #self.reg = ensemble.GradientBoostingRegressor(**params)
        '''
        self.reg = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                          
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
        
        self.reg = XGBRegressor(base_score=0.5, booster='gbtree', 
             colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1.0, gamma=0.6,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=7, min_child_weight=5, missing=None, n_estimators=1000,
             n_jobs=1, nthread=-1, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=0.6, verbosity=1) 
        '''
        self.reg = RandomForestRegressor()
        #self.reg = DecisionTreeRegressor(max_depth = 15)
        #self.reg = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
        #    kernel_params=None)
        #self.reg = LinearRegression()
    def fit(self, X, y):
        X_sc = self.scaler.fit_transform(X)
        self.reg.fit(X_sc, y)

    def predict(self, X):
        X_sc = self.scaler.transform(X)
        return self.reg.predict(X_sc)
```

```python
regressor_1 = Regressor()
regressor_2 = Regressor()
regressor_3 = Regressor()
regressor_4 = Regressor()
```

```python
regressor_1.fit(X, y1)
```

```python
regressor_2.fit(X,y2)
```

```python
regressor_3.fit(X,y3)
```

```python
regressor_4.fit(X,y4)
```

```python
pred_1 = regressor_1.predict(X_t)
pred_2 = regressor_2.predict(X_t)
pred_3 = regressor_3.predict(X_t)
```

```python
len(pred_2)
```

```python
def metric_nilm(dataframe_y_true, dataframe_y_pred):
    score = 0.0
    test = dataframe_y_true['washing_machine']
    pred = dataframe_y_pred['washing_machine']
    score += mt.sqrt(sum((pred.values - test.values)**2)/len(test))*5.55
    test = dataframe_y_true['fridge_freezer']
    pred = dataframe_y_pred['fridge_freezer']
    score += mt.sqrt(sum((pred.values - test.values)**2)/len(test))*49.79
    test = dataframe_y_true['TV']
    pred = dataframe_y_pred['TV']
    score += mt.sqrt(sum((pred.values - test.values)**2)/len(test))*14.57
    test = dataframe_y_true['kettle']
    pred = dataframe_y_pred['kettle']
    score += mt.sqrt(sum((pred.values - test.values)**2)/len(test))*4.95
    score /= 74.86
    return score
```

```python
metric_nilm(y_valid, pred)
```
