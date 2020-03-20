```python
from tensorflow import math as mt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, GRU, TimeDistributed, InputLayer, Conv1D, Conv2D, Conv3D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from x_pipeline import XPipeline
from y_pipeline import YPipeline

# from rnn_model import RNNModel
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
pd.set_option('display.max_columns', None)
```

## Preprocessing

```python
Y_train.drop(["washing_machine", "TV", "kettle"], axis=1, inplace=True)
```

```python
print('Start of first transform')
x = px.fit(X_train)
x = px.transform(X_train)
print('End of first transform')
y = py.fit(Y_train)
y = py.transform(Y_train)
print('Second transform')
x_train, y_train = x[:6000, :], y[:6000, :, :]
x_valid, y_valid = x[6000:, :], y[6000:, :, :]
```

```python
print(f"x_train shape is {x_train.shape}")
print(f"x_valid shape is {x_valid.shape}")
```

## Wave Net


Custom metric:

```python
@tf.function
def metric_nilm(y_true, y_pred):
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2]])
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]])
    score = 0.0

    test = tf.slice(y_true, [0, 0], [-1, 1])
    pred = tf.slice(y_pred, [0, 0], [-1, 1])
    score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 5.55

#     test = tf.slice(y_true, [0, 1], [-1, 1])
#     pred = tf.slice(y_pred, [0, 1], [-1, 1])
#     score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 49.79

#     test = tf.slice(y_true, [0, 2], [-1, 1])
#     pred = tf.slice(y_pred, [0, 2], [-1, 1])
#     score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 14.57

#     test = tf.slice(y_true, [0, 3], [-1, 1])
#     pred = tf.slice(y_pred, [0, 3], [-1, 1])
#     score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 4.95

    score /= 5.55
    
    return score
```

```python
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[60, 48]))
for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                  activation="relu", dilation_rate=rate))
model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
```

```python
model.compile(loss=metric_nilm, optimizer=Adam(lr=0.001))
history = model.fit(x_train, y_train, epochs=50,
                    validation_data=(x_valid, y_valid))
```

### Evaluating Performance

```python
COLUMNS = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']
```

```python
pred = model.predict(x_valid)
pred = pred.reshape((pred.shape[0] * pred.shape[1], pred.shape[2]))
pred = pd.DataFrame(pred, columns=COLUMNS)
```

```python
def nilm_metric(y_true, y_pred):
        if not isinstance(y_true, pd.DataFrame):
            y_true = y_true.reshape((y_true.shape[0] * y_true.shape[1], y_true.shape[2]))
            y_true_df = pd.DataFrame(y_true, columns=COLUMNS)
        else:
            y_true_df = y_true
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2]))
            y_pred_df = pd.DataFrame(y_pred, columns=COLUMNS)
        else:
            y_pred_df = y_pred

        score = 0.0
        test = y_true_df['washing_machine']
        pred = y_pred_df['washing_machine']
        score += math.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 5.55
        test = y_true_df['fridge_freezer']
        pred = y_pred_df['fridge_freezer']
        score += math.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 49.79
        test = y_true_df['TV']
        pred = y_pred_df['TV']
        score += math.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 14.57
        test = y_true_df['kettle']
        pred = y_pred_df['kettle']
        score += math.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 4.95
        score /= 74.86
        return score
```

```python
nilm_metric(y_valid, pred)
```

```python
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(pred.kettle, y_valid[...,3].flatten())
plt.plot(np.linspace(0,3000), np.linspace(0,3000), c="orange")
plt.show()
```

## GRU

```python
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[None, 48]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])
```

```python
model.compile(loss=metric_nilm, optimizer=Adam(lr=0.001))
history = model.fit(x_train, y_train, epochs=50,
                    validation_data=(x_valid, y_valid))
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
x_test = px.transform(X_test.iloc[:-1,])
```

```python
pred = model.predict(x_test)
pred = pred.reshape((pred.shape[0] * pred.shape[1], pred.shape[2]))
pred = pd.DataFrame(pred, columns=COLUMNS)
```

```python
pred = pd.concat([time, pred], axis=1)
```

```python
pred.fillna(method="ffill", inplace=True)
```

```python
pred.to_csv("test_submission.csv", index=False)
```
