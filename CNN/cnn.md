```python
from tensorflow import math as mt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, GRU, TimeDistributed, InputLayer, Conv1D, Conv2D, Conv3D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from x_pipeline import XPipeline
from y_pipeline import YPipeline
```

```python
X_train = pd.read_csv(
    '../provided_data_and_metric/X_train_6GWGSxz.csv',
)
X_train.set_index("time_step", inplace=True)
Y_train = pd.read_csv(
    '../provided_data_and_metric/y_train_2G60rOL.csv',
)
Y_train.set_index("time_step", inplace=True)
X_test = pd.read_csv(
    '../provided_data_and_metric/X_test_c2uBt2s.csv',
)
X_test.set_index("time_step", inplace=True)

px = XPipeline()
py = YPipeline()
```

```python
X_train.columns
```

```python
pd.set_option('display.max_columns', None)
```

## Preprocessing

```python
print('Start of first transform')
x = px.fit(X_train)
x = px.transform(X_train)
print('End of first transform')
y = py.fit(Y_train)
y = py.transform(Y_train)
print('Second transform')
x_train, y_train = x[:321000, :, :], y[:321000]
x_valid, y_valid = x[321000:, :, :], y[321000:]
```

```python
x.shape
```

```python
print(f"x_train shape is {x_train.shape}")
print(f"x_valid shape is {x_valid.shape}")
```

## Wave Net

```python
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[256, 1]))
for k, f in zip((5, 4, 3, 2), (20, 20, 20, 20)):
    model.add(keras.layers.Conv1D(filters=f, kernel_size=k, padding="valid",
                                  activation="relu"))
model.add(Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
```

```python
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[256, 1]))

model.add(keras.layers.Conv1D(filters=30, kernel_size=5, padding="valid",
                              activation="relu"))
model.add(keras.layers.Conv1D(filters=30, kernel_size=4, padding="valid",
                              activation="relu"))
model.add(keras.layers.Conv1D(filters=40, kernel_size=3, padding="valid",
                              activation="relu"))
model.add(keras.layers.Conv1D(filters=50, kernel_size=2, padding="valid",
                              activation="relu"))
model.add(keras.layers.Conv1D(filters=50, kernel_size=2, padding="valid",
                              activation="relu"))
model.add(Flatten())
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
```

```python
model.summary()
```

Metric: Mean Squared Error. Same as minimizing the Metric on the platform.

```python
model.compile(loss=keras.losses.mean_squared_error, optimizer=Adam())
history = model.fit(x_train, y_train, epochs=3,
                    validation_data=(x_valid, y_valid))
```

```python
pred_val = model.predict(x_valid)
```

```python
pred_val[pred_val<0] = 0
```

### Evaluating Performance

```python
def nilm_metric(y_true, y_pred):
        score = 0.0
        score += math.sqrt(sum((y_pred - y_true) ** 2) / len(y_true)) * 49.79
        score /= 74.86
        return score
```

```python
nilm_metric(y_valid.values, pred_val)
```

```python
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(pred_val, y_valid)
plt.plot(np.linspace(0,250), np.linspace(0,250), c="orange")
plt.show()
```

### submission

```python
X_test = pd.read_csv(
    '../provided_data_and_metric/X_test_c2uBt2s.csv',
)
time = X_test["time_step"]
X_test.set_index("time_step", inplace=True)
```

```python
x_test = px.transform(X_test)
```

```python
pred = model.predict(x_test)
pred[pred<0] = 0
pred = pd.DataFrame(pred, columns=["fridge_freezer"])
```

```python
pred = pd.concat([time, pred], axis=1)
```

```python
pred.to_csv("fridge_freezer.csv", index=False)
```
