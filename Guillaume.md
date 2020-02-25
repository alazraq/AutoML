---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from keras.layers import LSTM, Dense
from keras.models import Sequential
```

```python
x_train = pd.read_csv(
    'provided_data_and_metric/X_train_6GWGSxz.csv',
     index_col=0
)
y_train = pd.read_csv(
    'provided_data_and_metric/y_train_2G60rOL.csv',
     index_col=0 
)
x_test = pd.read_csv(
    'provided_data_and_metric/X_test_c2uBt2s.csv', 
     index_col=0 
)
```

```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
```

```python
x_train.shape[0] - x_train.isna().sum() #/ x_train.shape[0]
```

```python
na_per_row = x_train['consumption'].isna()
x_train['consumption'][na_per_row == True].index
```

```python
y_train.isna().sum() / x_train.shape[0]
```

```python
na_per_row = y_train.isna().sum(axis = 1)
print(na_per_row[na_per_row != 0])
```

```python
plt.plot(x_train.iloc[1:1000, :]['consumption'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.xlabel(' ')
plt.show()
```

```python
d
```

```python
class DataImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X = X.interpolate(method='linear').fillna(method='bfill')
        X.index = pd.to_datetime(X.index)
        X['hour'] = X.index.map(lambda x: x.hour)
        X['minute'] = X.index.map(lambda x: x.minute)
        X['day'] = X.index.map(lambda x: x.day)
        X['month'] = X.index.map(lambda x: x.month)
        X['weekday'] = X.index.map(lambda x: x.weekday)
        try:
            X.drop('Unnamed: 9', axis = 1, inplace = True)
        except KeyError as e:
            pass
        return X
```

```python
class YImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X = X.interpolate(method='linear').fillna(method='bfill')
        X.index = pd.to_datetime(X.index)
        return X
```

```python
class NeuralNet(BaseEstimator):
    
    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(
            units=64
        ))
        self.model.add(Dense(units=4))
        self.model.compile()
    
    def fit(self, X, y=None):
        self.model.fit(X)
        
    def predict(self, y):
        return self.model.predict(y)
```

```python
di = DataImputer()
yi = YImputer()
x_trans = di.transform(X=x_train)
y_trans = yi.transform(X=y_train)
```

```python
rf = RandomForestRegressor()
rf.fit(x_trans, y_trans)
```

```python
pred_df = pd.DataFrame(rf.predict(di.transform(x_train)), index=x_train.index, columns=y_train.columns)
```

```python
pred_df.to_csv('./provided_data_and_metric/y_pred.csv')
```

```python
# We check that there are no more NaN after transformation.
transformed_x.isna().sum()
```

```python
p.transform(y_train)
```

```python

```

```python

```

```python
y_train_copy = y_train.copy()
```

```python
def roundTime(dt=None, roundTo=60):
   """
   Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)
```

```python
y_train_copy['time_step'] = y_train_copy['time_step'].apply(
    lambda x : roundTime(datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f")).hour
)

```

```python
y_train_copy['time_step'].unique()
```

```python
for label in y_train_copy.columns[1:]:
    print(label)
    plt.scatter(y_train_copy['time_step'], y_train_copy[label], label = label)

plt.legend()
plt.show()
```

```python
for label in y_train.columns[1:]:
    print(label)
    plt.scatter(x_train['consumption'], y_train[label], label = label)

plt.legend()
plt.show()
```

```python
print('This is a test for jupyText, this should work,and if I add some content live too.')
```

```python
class RNNModel():
    def __init__(self, hidden_neurons, nb_columns_X, nb_columns_Y):
        self.model = Sequential()
        self.model.add(
            LSTM(
                hidden_neurons, 
                batch_input_shape=(1, 1, nb_columns_X), 
                return_sequences=False,
                stateful=True
            )
        )
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_columns_Y))
        self.model.add(Activation("linear"))
        self.model.compile(
            loss='mean_squared_error', 
            optimizer="rmsprop",
            metrics=['mean_squared_error']
        )
        
    def fit(self, X, y):
        di = DataImputer()
        yi = YImputer()
        x_trans = di.transform(X=x_train)
        y_trans = yi.transform(X=y_train)
        self.model.fit(x_trans, y_trans)
        

nb_columns_Y = y_train.shape[2]
nb_columns_X = X_train.shape[2]
```

```python
import zoo
from zoo.common.nncontext import *
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.keras.layers import *

# Get the current Analytics Zoo version
zoo.__version__
# Create a SparkContext and initialize the BigDL engine.
sc = init_nncontext()
# Create a Sequential model containing a Dense layer.
model = Sequential()
model.add(Dense(8, input_shape=(10, )))
```
