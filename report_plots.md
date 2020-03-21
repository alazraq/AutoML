```python
import pandas as pd
# import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import holidays
import math as mt

```

```python
X_train = pd.read_csv(
    'provided_data_and_metric/X_train_6GWGSxz.csv',
)
X_train.set_index("time_step", inplace=True)
X_train.index = pd.to_datetime(X_train.index)
Y_train = pd.read_csv(
    'provided_data_and_metric/y_train_2G60rOL.csv',
)
Y_train.set_index("time_step", inplace=True)
Y_train.index = pd.to_datetime(Y_train.index)
```

```python
X_weekly = X_train.iloc[:, :1].resample('W').mean()
Y_weekly = Y_train.resample('W').mean()
T_weekly = X_weekly.join(Y_weekly, on='time_step')

f,ax = plt.subplots(figsize=(15,8))

ax.plot(T_weekly.index, T_weekly[['consumption']], label='consumption', color='black')
ax.stackplot(
    T_weekly.index.values,
    T_weekly[['washing_machine', 'fridge_freezer', 'TV', 'kettle']].values.T,
    labels=T_weekly.columns[1:]
)
ax.set_ylim([0, 700])
ax.legend(loc='upper right')

```

```python
X_daily = X_train.iloc[:, :1].resample('D').mean()
Y_daily = Y_train.resample('D').mean()
X_daily['mv_consumption'] = X_daily['consumption'].rolling(7).mean()
Y_daily = Y_daily.rolling(7).mean()
T_daily = X_daily.join(Y_daily, on='time_step')

f,ax = plt.subplots(figsize=(15,8))

ax.plot(T_daily.index, T_daily[['mv_consumption']], label='consumption', color='black')
ax.stackplot(
    T_daily.index.values,
    T_daily[['washing_machine', 'fridge_freezer', 'TV', 'kettle']].values.T,
    labels=T_daily.columns[1:]
)
ax.set_ylim([0, 700])
ax.legend(loc='upper left')

```
