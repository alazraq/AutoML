---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
---

```python
import pandas as pd
from x_pipeline import XPipeline
from y_pipeline import YPipeline

# from rnn_model import RNNModel
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
X_test.drop('Unnamed: 9', axis=1, inplace=True)
X_test.set_index("time_step", inplace=True)

p1 = XPipeline()
p2 = YPipeline()
```

```python
print('Start of first transform')
x = p1.fit(X_train)
x = p1.transform(X_train)
print('End of first transform')
y = p2.fit(Y_train)
y = p2.transform(Y_train)
print('Second transform')
x_train, y_train = x[6000:, :], y[6000:, :, :]
x_valid, y_valid = x[:6000, :], y[:6000, :, :]
```

```python
print(f"x_train shape is {x_train.shape}")
print(f"x_valid shape is {x_valid.shape}")
```

```python
import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, SimpleRNN, TimeDistributed
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pandas as pd
import math
from tensorflow import math as mt

class RNNModel:

    COLUMNS = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']

    def __init__(self, batch_size=60, features=17):
        self.batch_size = batch_size
        self.features = features
        self.model = Sequential()
#         self.model.add(SimpleRNN(
#             units=20,
#             input_shape=[self.batch_size, self.features], 
#             return_sequences=True
#         ))
        self.model.add(LSTM(
            units=60,
            return_sequences=True,
            input_shape=(self.batch_size, self.features)
        ))
#         self.model.add(LSTM(60, return_sequences=True))  
#         self.model.add(LSTM(19))
        self.model.add(Dense(units=4))
        
        self.callbacks = []
        self.callbacks.append(EarlyStopping(monitor='accuracy', patience=2))
        self.model.compile(
            loss=self.metric_nilm, 
            optimizer="adam", 
            metrics=['accuracy']
        )
        self.history = None

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, epochs=10):
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
#             validation_data=(x_valid, y_valid),
            callbacks=self.callbacks
        )

    def predict(self, x_test):
#         pred = self.model.predict(x_test)
#         pred = pred.reshape((pred.shape[0] * pred.shape[1], pred.shape[2]))
#         return pd.DataFrame(pred, columns=RNNModel.COLUMNS)
        return self.model.predict(x_test)

    @tf.function
    def metric_nilm(self, y_true, y_pred):
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0] * tf.shape(y_pred)[1], tf.shape(y_pred)[2]])
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0] * tf.shape(y_true)[1], tf.shape(y_true)[2]])
        score = 0.0

        test = tf.slice(y_true, [0, 0], [-1, 1])
        pred = tf.slice(y_pred, [0, 0], [-1, 1])
        score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 5.55

        test = tf.slice(y_true, [0, 1], [-1, 1])
        pred = tf.slice(y_pred, [0, 1], [-1, 1])
        score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 49.79

        test = tf.slice(y_true, [0, 2], [-1, 1])
        pred = tf.slice(y_pred, [0, 2], [-1, 1])
        score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 14.57

        test = tf.slice(y_true, [0, 3], [-1, 1])
        pred = tf.slice(y_pred, [0, 3], [-1, 1])
        score += mt.sqrt(mt.reduce_sum(mt.subtract(pred, test) ** 2) / float(len(test))) * 4.95

        score /= 74.86

        return score
```

```python
rnn = RNNModel(60, x_train.shape[2])
rnn.fit(x_train, y_train, x_valid, y_valid, 30)
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
import pandas as pd
val_pred = rnn.predict(x_valid)
# y_valid = y_valid.reshape((y_valid.shape[0] * y_valid.shape[1], y_valid.shape[2]))
print(y_valid.shape)
print(val_pred.shape)
nilm_metric(y_valid, val_pred)

```
