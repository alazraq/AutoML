---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Data Challenge: Smart meter is coming
by BCM Energy - Planète OUI

```python
import pandas as pd
# import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import holidays
import math as mt

import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, SimpleRNN
from keras.models import Sequential

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
```

```python
X_train.head()
```

## Preprocessing


Dealing with NaNs:

```python
class DataImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        try:
            X.drop('Unnamed: 9', axis = 1, inplace = True)
        except KeyError as e:
            pass
        X = X.interpolate(method='linear').fillna(method='bfill')
        X.time_step = pd.to_datetime(X.time_step)
        return X
```

```python
class YImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X = X.interpolate(method='linear').fillna(method='bfill')
#         X.index = pd.to_datetime(X.index)
        return X
```

Adding Extra Features:

```python
class DataAugmenter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        fr_holidays = holidays.France()
        X["weekday"] = X.time_step.dt.dayofweek
        X["month"] = X.time_step.dt.month
        X["hour"] = X.time_step.dt.hour
        X["is_weekend"] = (X["weekday"] > 4)*1  
        X["is_holidays"] = (X.time_step.apply(lambda x: x in(fr_holidays)))*1
        
        X["is_breakfast"] = ((X.hour>5) & (X.hour<9))*1 
        X["is_teatime"] = ((X.hour>16) & (X.hour<20))*1 
        X["is_TVtime"] = ((X.hour>17) & (X.hour<23))*1
        # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
        X["is_night"] = ((X.hour>0) & (X.hour<7))*1
        return X
```

Building a custom OneHotEncoder:

```python
class MyOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.all_possible_hours = np.arange(0, 24)
        self.all_possible_weekdays = np.arange(0, 7)
        self.all_possible_months = np.arange(1, 13)
        self.ohe_hours = OneHotEncoder(drop="first")
        self.ohe_weekdays = OneHotEncoder(drop="first")
        self.ohe_months = OneHotEncoder(drop="first")
    
    def fit(self, X, y=None):
        self.ohe_hours.fit(self.all_possible_hours.reshape(-1,1))
        self.ohe_weekdays.fit(self.all_possible_weekdays.reshape(-1,1))
        self.ohe_months.fit(self.all_possible_months.reshape(-1,1))
        return self

    def transform(self, X, y=None):
#         self.fit(X)
        hours = pd.DataFrame(self.ohe_hours.transform(X.hour.values.reshape(-1,1)).toarray(), 
                             columns=["hour_"+str(i) for i in range(1, 24)])
        weekdays = pd.DataFrame(self.ohe_weekdays.transform(X.weekday.values.reshape(-1,1)).toarray(), 
                             columns=["weekday_"+str(i) for i in range(1, 7)])
        months = pd.DataFrame(self.ohe_months.transform(X.month.values.reshape(-1,1)).toarray(), 
                             columns=["month_"+str(i) for i in range(2, 13)])
        X = pd.concat([X, hours, weekdays, months], axis=1)
        return X
```

## Modeling


### Preprocessing

```python
pd.set_option('display.max_columns', None)
```

Testing Pipeline:

```python
X_train = pd.read_csv(
    'provided_data_and_metric/X_train_6GWGSxz.csv',
)
Y_train = pd.read_csv(
    'provided_data_and_metric/y_train_2G60rOL.csv',
)
```

It doesn't work if I uncomment. Why?

```python
p = Pipeline([
    (
        '1',
        DataImputer()
    ),
    (
        '2',
        DataAugmenter()
    ),
    (
        '3',
        MyOneHotEncoder()
    ),
])
```

```python
p.fit(X_train)
```

```python
data = p.transform(X_train)
```

```python
ti = YImputer()
target = ti.transform(Y_train)
```

```python
x_train, x_valid, y_train, y_valid = train_test_split(
    data.drop('time_step', axis=1), target.drop('time_step', axis=1), test_size=0.30, random_state=42)
```

### Regression, a baseline model


[MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) consists of fitting one regressor per target. 

This is a simple strategy for extending regressors that do not natively support multi-target regression.


Test cold months only:

```python
# Y_train = Y_train.loc[(X_train.month<4) | (X_train.month>8)]
# X_train = X_train.loc[(X_train.month<4) | (X_train.month>8)]
```

```python
regressor = MultiOutputRegressor(LinearRegression())
```

```python
regressor.fit(x_train, y_train)
```

```python
regressor.score(x_train, y_train)
```

```python
regressor.score(x_valid, y_valid)
```

Have a look at all the coefficients:

```python
df = pd.DataFrame([i.coef_ for i in regressor.estimators_], columns=x_train.columns, index=y_train.columns)
```

```python
df
```

What about predictions?

```python
y_pred = regressor.predict(x_valid)
```

```python
pred = pd.DataFrame(y_pred, columns=y_train.columns)
pred["time_step"] = X_train.time_step
```

```python
pred.head()
```

Metric used on the website:

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

##### Test Submission

```python
X_test = pd.read_csv(
    'provided_data_and_metric/X_test_c2uBt2s.csv', 
)
```

Save time for later

```python
time = X_test["time_step"]
```

```python
x_test = p.transform(X_test)
```

```python
x_test.head()
```

```python
y_pred = regressor.predict(x_test.iloc[:,1:])
```

```python
x_test.iloc[:,1:]
```

```python
pred = pd.DataFrame(y_pred, columns=y_train.columns)
pred = pd.concat([time, pred], axis=1)
```

```python
pred.head()
```

```python
pred.to_csv("test_submission.csv", index=False)
```
