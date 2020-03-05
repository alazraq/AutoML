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
```

```python
x_train = pd.read_csv('provided_data_and_metric/X_train_6GWGSxz.csv')
y_train = pd.read_csv('provided_data_and_metric/y_train_2G60rOL.csv')
x_test = pd.read_csv('provided_data_and_metric/X_test_c2uBt2s.csv')
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
y_train[610:630].interpolate(method='linear')
```

```python
from keras.layers import LSTM, Dense
from keras.models import Sequential
```

```python
import datetime
import matplotlib.pyplot as plt
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
