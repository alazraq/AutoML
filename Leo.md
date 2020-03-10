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

## Data Exploration

```python
X_train.head()
```

### Dealing with NaN

```python
X_train.consumption[X_train.consumption.isna()]
```

DataImputer and YImputer are custom trasformers we have built to deal with NaNs.

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

```python
di = DataImputer()
yi = YImputer()
X_train = di.transform(X=X_train)
Y_train = yi.transform(X=Y_train)
```

```python
X_train.head()
```

```python
# X_train.consumption.fillna(method="ffill", inplace=True)
# X_test.consumption.fillna(method="ffill", inplace=True)
# Y_train.fillna(method="ffill", inplace=True)
```

```python
X_train["time_step"] = pd.to_datetime(X_train["time_step"])
```

```python
fr_holidays = holidays.France()
```

DO NOT DELETE THIS PLEASE OR WE DO NOT GET IS_HOLIDAYS WORKING!

```python
c = 0
for i in X_train.time_step.dt.date:
    if i in fr_holidays:
        c+=1
        
print(c)
```

Adding extra features:


Do we also want to add day of the month?

```python
X_train["weekday"] = X_train.time_step.dt.dayofweek
X_train["month"] = X_train.time_step.dt.month
X_train["hour"] = X_train.time_step.dt.hour
X_train["is_weekend"] = (X_train["weekday"] > 4)*1  
X_train["is_holidays"] = (X_train.time_step.dt.date.isin(fr_holidays))*1
```

### Visualizing the Data


There is on average more consumption during weekends, as expected.

```python
X_train[["consumption", "is_weekend"]].groupby("is_weekend").mean()
```

---


Weekday:

```python
X_train[["consumption", "weekday"]].groupby("weekday").mean()
```

```python
sns.lineplot(x=np.arange(0,7), y="consumption", data=X_train.groupby("weekday").mean())
```

month:

```python
X_train[["consumption", "month"]].groupby("month").mean()
```

Significant drop in consumption over the summer! We do not have data for January, February. 

```python
sns.lineplot(x=np.arange(3,13), y="consumption", data=X_train.groupby("month").mean())
```

In the afternoon, the most consumption.

```python
sns.lineplot(x=np.arange(0,24), y="consumption", data=X_train.groupby("hour").mean())
```

Holidays:

```python
X_train.consumption.std()
```

```python
X_train[["consumption", "is_holidays"]].groupby("is_holidays").mean()
```

Due to the big difference in consumption, it looks like the data belongs to a city in France.

```python
X_train[["consumption", "is_holidays"]].groupby("is_holidays").std()
```

Plots:

```python
# fig, axs = plt.subplots(2,2, figsize=(15,15))
# axs[0,0].scatter(X_train.consumption[Y_train.washing_machine > 0], Y_train.washing_machine[Y_train.washing_machine > 0], c="red")
# axs[0,0].scatter(X_train.consumption[Y_train.washing_machine == 0], Y_train.washing_machine[Y_train.washing_machine == 0], c="blue")
# axs[0,1].scatter(X_train.consumption, Y_train.fridge_freezer )
# axs[1,0].scatter(X_train.consumption, Y_train.TV )
# axs[1,1].scatter(X_train.consumption, Y_train.kettle)
```

<!-- #region cell_style="center" -->
### Analyzing the target
<!-- #endregion -->

Weekday:
- People enjoy using the Washing Machine on Sunday

```python
Y_train.groupby(X_train.weekday).mean()
```

Month:
- Significant increase in the use of the Washing Machine and the Kettle in November

```python
Y_train.groupby(X_train.month).mean()
```

Weekend:

```python
Y_train.groupby(X_train.is_weekend).mean()
```

Hour:
- Washing Machine used late evening
- TV from the evening
- Kettle in the afternoon around Tea Time

```python
Y_train.groupby(X_train.hour).mean()
```

Holidays:
- Who wants to do a Washing Machine while on holidays?

```python
Y_train.groupby(X_train.is_holidays).mean()
```

### Adding extra features


By looking at the data, these features can be added:

```python
X_train["is_breakfast"] = ((X_train.hour>5) & (X_train.hour<9))*1 
X_train["is_teatime"] = ((X_train.hour>16) & (X_train.hour<20))*1 
X_train["is_TVtime"] = ((X_train.hour>17) & (X_train.hour<23))*1
# X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
X_train["is_night"] = ((X_train.hour>0) & (X_train.hour<7))*1
```

Let's put all in a transformer!

```python
class DataAugmenter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X["time_step"] = pd.to_datetime(X["time_step"])
        X["weekday"] = X.time_step.dt.dayofweek
        X["month"] = X.time_step.dt.month
        X["hour"] = X.time_step.dt.hour
        X["is_weekend"] = (X["weekday"] > 4)*1  
        X["is_holidays"] = (X.time_step.dt.date.isin(fr_holidays))*1
        
        X["is_breakfast"] = ((X.hour>5) & (X.hour<9))*1 
        X["is_teatime"] = ((X.hour>16) & (X.hour<20))*1 
        X["is_TVtime"] = ((X.hour>17) & (X.hour<23))*1
        # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
        X["is_night"] = ((X.hour>0) & (X.hour<7))*1
        return X
```

---
---
---


## Modeling


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
Y_train.head()
```

```python
x_train, x_valid, y_train, y_valid = train_test_split(
    X_train.drop('time_step', axis=1), Y_train.drop('time_step', axis=1), test_size=0.33, random_state=42)
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
df = pd.DataFrame([i.coef_ for i in regressor.estimators_], columns=X_train.columns[1:], index=Y_train.columns[1:])
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
X_test.drop('Unnamed: 9', axis = 1, inplace = True)
```

Save time for later

```python
time = X_test["time_step"]
```

```python
X_test = di.transform(X=X_test)
ag = DataAugmenter()
X_test = ag.transform(X=X_test)
```

```python
X_test.head()
```

```python
y_pred = regressor.predict(X_test.iloc[:,1:])
```

```python
pred = pd.DataFrame(y_pred, columns=Y_train.columns[1:])
pred= pd.concat([time, pred], axis=1)
```

```python
pred.head()
```

```python
pred.to_csv("test_submission.csv", index=False)
```

### Random Forest, a look at feature importance

```python
from sklearn.ensemble import RandomForestRegressor
```

```python
regressor = MultiOutputRegressor(RandomForestRegressor())
```

```python
x_train, x_valid, y_train, y_valid = train_test_split(
    X_train.drop('time_step', axis=1), Y_train.drop('time_step', axis=1), test_size=0.33, random_state=42)
```

```python
# regressor.fit(x_train, y_train)
```

```python
# print(regr.feature_importances_)
```

### Preprocessing


Build a custom OneHotEncoder

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

```python
oh = MyOneHotEncoder()
oh.fit(X_train)
```

```python
pd.set_option('display.max_columns', None)
```

```python
x_train = oh.transform(X_train)
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
p.transform(X_train)
```

## Predicting if appliance is on or off
There is a huge difference in consumption when an appliance is on or off.


Fridge:

```python
Y_train.fridge_freezer[Y_train.fridge_freezer != 0].mean()
```

```python
Y_train.fridge_freezer.mean()
```

Kettle:

```python
Y_train.kettle[Y_train.kettle != 0].mean()
```

```python
Y_train.kettle.mean()
```

Washing Machine:

```python
Y_train.washing_machine[Y_train.washing_machine != 0].mean()
```

```python
Y_train.washing_machine.mean()
```

The TV is always on:

```python
(Y_train.TV != 0).sum() == len(Y_train)
```

```python
Y_train.TV.std()
```

```python
fig, axs = plt.subplots(2,2, figsize=(15,15))
axs[0,0].scatter(X_train.consumption[Y_train.washing_machine > 0], Y_train.washing_machine[Y_train.washing_machine > 0], c="red")
axs[0,0].scatter(X_train.consumption[Y_train.washing_machine == 0], Y_train.washing_machine[Y_train.washing_machine == 0], c="blue")
axs[0,1].scatter(X_train.consumption, Y_train.fridge_freezer )
axs[1,0].scatter(X_train.consumption, Y_train.TV )
axs[1,1].scatter(X_train.consumption, Y_train.kettle)
```

---
---
---
---
---
---
---


## RNN, testing

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
X_train.set_index("time_step", inplace=True)
Y_train.set_index("time_step", inplace=True)
```

```python
X_train = X_train.append(X_train.iloc[-1, :])
Y_train = Y_train.append(Y_train.iloc[-1, :])
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
        X.index = pd.to_datetime(X.index)
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

```python
class DataAugmenter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X["time_step"] = pd.to_datetime(X["time_step"])
        X["weekday"] = X.time_step.dt.dayofweek
        X["month"] = X.time_step.dt.month
        X["hour"] = X.time_step.dt.hour
        X["is_weekend"] = (X["weekday"] > 4)*1  
        X["is_holidays"] = (X.time_step.dt.date.isin(fr_holidays))*1
        
        X["is_breakfast"] = ((X.hour>5) & (X.hour<9))*1 
        X["is_teatime"] = ((X.hour>16) & (X.hour<20))*1 
        X["is_TVtime"] = ((X.hour>17) & (X.hour<23))*1
        # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
        X["is_night"] = ((X.hour>0) & (X.hour<7))*1
        return X
```

```python
class RNNDataFormatter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X.set_index("time_step", inplace=True)
        X = X.append(X.iloc[-1, :])
        nb_col = X.shape[1]
        return X.values.reshape((int(X_rnn.shape[0]/60), 60, nb_col))
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
    ),
    (
        '3',
        RNNDataFormatter()
    )
])
```

```python
p2 = Pipeline([
    (
        '1',
        YImputer()
    ),
    (
        '2',
        RNNDataFormatter()
    )
])
```

```python
X = p1.transform(X_train)
```

```python
Y = p2.transform(Y_train)
```

```python
X.shape
```

```python
Y.shape
```

```python
x_train, y_train = X[:6000, :], Y[:6000, :, :]
x_valid, y_valid = X[6000:, :], Y[6000:, :, :]
```

### Baseline Model

```python
np.random.seed(42)
tf.random.set_seed(42)

model = Sequential([
    SimpleRNN(20, return_sequences=True, input_shape=[None, 14]),
    SimpleRNN(20, return_sequences=True),
    SimpleRNN(4, return_sequences=True)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_valid, y_valid))
```

```python
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
#     plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()
```

```python
model.predict(x_valid).shape
```

### Improvements


**TO-DO** (before running code below):
- Add ColumnTransformer, to avoid scaling categorical features.
- Add DataAugmenter in the pipeline
- Check if RNNDataFormatter still works


**Issue**: The RNN returns output from sigmoid, hence between (-1, 1).

**Idea**: Normalization


Modify RNNDataFormatter to make it work with np.arrays:

```python
class RNNDataFormatter(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return X
    
    def transform(self, X, y=None):
        X.set_index("time_step", inplace=True)
        X = X.append(X.iloc[-1, :])
        nb_col = X.shape[1]
        X = X.reshape((int(X.shape[0]/60), 60, nb_col))
        return X
```

Fit StandardScaler before, gives error in Pipeline

```python
scaler_x = StandardScaler()
scaler_x.fit(X_train)
```

```python
scaler_y = StandardScaler()
scaler_y.fit(Y_train)
```

```python
p1 = Pipeline([
        (
        '1',
        DataImputer()
    ),
    (
        '2',
        scaler_x
    ),
    (
        '3',
        RNNDataFormatter()
    )
])
```

```python
p2 = Pipeline([
    (
        '1',
        YImputer()
    ),
    (
        '2',
        scaler_y
    ),
    (
        '3',
        RNNDataFormatter()
    )
])
```

```python
x_scaled = p1.transform(X_train)
y_scaled = p2.transform(Y_train)
```

```python
print(f"x_scaled shape is {x_scaled.shape}")
print(f"y_scaled shape is {y_scaled.shape}")
```

```python
x_train, y_train = x_scaled[:6000, :], y_scaled[:6000, :, :]
x_valid, y_valid = x_scaled[6000:, :], y_scaled[6000:, :, :]
```

```python
np.random.seed(42)
tf.random.set_seed(42)

model = Sequential([
    SimpleRNN(20, return_sequences=True, input_shape=[None, 8]),
    SimpleRNN(20, return_sequences=True),
    SimpleRNN(4, return_sequences=True)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_valid, y_valid))
```

```python
y_pred = model.predict(x_train)
```

```python
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 4)) #shape is now (360000, 4)
```

```python
mean_squared_error(y_trans.iloc[:360000, :], y_pred)
```

```python
y_trans.head()
```

```python
y_pred[:5, :]
```
