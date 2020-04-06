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
#         X['hour'] = X.index.map(lambda x: x.hour)
#         X['minute'] = X.index.map(lambda x: x.minute)
#         X['day'] = X.index.map(lambda x: x.day)
#         X['month'] = X.index.map(lambda x: x.month)
#         X['weekday'] = X.index.map(lambda x: x.weekday)
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
from keras.layers import Dropout, Activation
class RNNModel():
    def __init__(self, hidden_neurons, nb_columns_X, nb_columns_Y):
        self.model = Sequential()
        self.model.add(
            LSTM(
                hidden_neurons, 
                batch_input_shape=(1, 60, nb_columns_X-1),  # TO DO: change
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
    def fit(self, X, y, epochs = 2, batch_size = 60):
        di = DataImputer()
        rdf = RNNDataFormatter()
        yi = YImputer()
        x_trans = rdf.transform(di.transform(X=x_train))
        X = x_trans
        y_trans = yi.transform(X=y_train)
        y = rdf.transform(y_trans)
        max_length = x_trans.shape[1]
        for epoch in range(epochs):
            mean_tr_acc = []
            mean_tr_loss = []
            for s in range(X.shape[0]): 
                X_st = X[s,:,:]
                y_st = y[s,:,:]
                print('gg')
                X_st.reshape(1,60,8)
                print(X_st.shape)
                tr_loss, tr_acc = self.model.train_on_batch([X_st],y_st)
                '''
                for t in range(max_length):
                    X_st = X[s][t]
                    y_st = y[s][t]
                    tr_loss, tr_acc = self.model.train_on_batch(X_st,y_st)
                    '''
                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
            model.reset_states()
            print('<loss (mse) training> {}'.format(np.mean(mean_tr_loss)))

model = RNNModel(8, x_train.shape[1], y_train.shape[1])
model.fit(x_train, y_train)
```

```python
# import zoo
# from zoo.common.nncontext import *
# from zoo.pipeline.api.keras.models import *
# from zoo.pipeline.api.keras.layers import *

# # Get the current Analytics Zoo version
# zoo.__version__
# # Create a SparkContext and initialize the BigDL engine.
# sc = init_nncontext()
# # Create a Sequential model containing a Dense layer.
# model = Sequential()
# model.add(Dense(8, input_shape=(10, )))
```

```python
import numpy as np
def random_sample(len_timeseries=3000):
    Nchoice = 600
    x1 = np.cos(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))
    x2 = np.cos(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))
    x3 = np.sin(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))
    x4 = np.sin(np.arange(0,len_timeseries)/float(1.0 + np.random.choice(Nchoice)))
    y1 = np.random.random(len_timeseries)
    y2 = np.random.random(len_timeseries)
    y3 = np.random.random(len_timeseries)
    for t in range(3,len_timeseries):
        ## the output time series depend on input as follows: 
        y1[t] = x1[t-2] 
        y2[t] = x2[t-1]*x3[t-2]
        y3[t] = x4[t-3]
    y = np.array([y1,y2,y3]).T
    X = np.array([x1,x2,x3,x4]).T
    return y, X
def generate_data(Nsequence = 1000):
    X_train = []
    y_train = []
    for isequence in range(Nsequence):
        y, X = random_sample(10)
        X_train.append(X)
        y_train.append(y)
    return np.array(X_train),np.array(y_train)

X, y = generate_data(2)
X
```

```python
class RNNDataFormatter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        print(X.shape)
        nb_col = X.shape[1]
        print(nb_col)
        X_rnn = X[X.columns[:nb_col]].iloc[0:18000]
        X_rnn = X_rnn.values.reshape((int(X_rnn.shape[0]/60), 60, nb_col))
        X_rnn[0, :, :]
        return X_rnn
```

```python
p = Pipeline([
    (
        '1',
        DataImputer()
    ),
    (
        '2',
        RNNDataFormatter()
    )
])
```

```python
p.transform(x_train)[0, 0, :]
```
