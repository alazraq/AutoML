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
x_train, y_train = x[:6000, :], y[:6000, :, :]
x_valid, y_valid = x[6000:, :], y[6000:, :, :]
```

```python
print(x_train.shape)
print(x_valid.shape)
```

```python
print(x_train.shape)
```

```python
import tensorflow as tf
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
tf.slice(t, [1, 0, 0], [3, 1, 2])  # [[[3, 3, 3]]]

```

```python
from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, SimpleRNN, TimeDistributed
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import pandas as pd
import math as mt

def nilm_metric(y_true, y_pred):
    print(y_true)
    y_true_df = None
    y_pred_df = None
    if not isinstance(y_true, pd.DataFrame):
        y_true_df = pd.DataFrame(y_true, columns=RNNModel.COLUMNS)
    else:
        y_true_df = y_true
    if not isinstance(y_pred, pd.DataFrame):
        y_pred_df = pd.DataFrame(y_pred, columns=RNNModel.COLUMNS)
    else:
        y_pred_df = y_pred

    score = 0.0
    test = y_true_df['washing_machine']
    pred = y_pred_df['washing_machine']
    score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 5.55
    test = y_true_df['fridge_freezer']
    pred = y_pred_df['fridge_freezer']
    score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 49.79
    test = y_true_df['TV']
    pred = y_pred_df['TV']
    score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 14.57
    test = y_true_df['kettle']
    pred = y_pred_df['kettle']
    score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 4.95
    score /= 74.86
    return score

def nilm_metric(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    coeffs = [5.55, 49.79, 14.57, 4.95]
    for col in y_true.shape[2]:
        pass
    return 0

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
        self.model.add(LSTM(60, return_sequences=True))  
#         self.model.add(LSTM(19))
        self.model.add(Dense(units=4))
        
        self.callbacks = []
        self.callbacks.append(EarlyStopping(monitor='accuracy', patience=2))
        self.model.compile(
            loss=nilm_metric, 
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
        pred = self.model.predict(x_test)
        pred = pred.reshape((pred.shape[0] * pred.shape[1], pred.shape[2]))
        return pd.DataFrame(pred, columns=RNNModel.COLUMNS)

#     def nilm_metric(self, y_true, y_pred):
#         y_true_df = None
#         y_pred_df = None
#         if not isinstance(y_true, pd.DataFrame):
#             y_true_df = pd.DataFrame(y_true, columns=RNNModel.COLUMNS)
#         else:
#             y_true_df = y_true
#         if not isinstance(y_pred, pd.DataFrame):
#             y_pred_df = pd.DataFrame(y_pred, columns=RNNModel.COLUMNS)
#         else:
#             y_pred_df = y_pred

#         score = 0.0
#         test = y_true_df['washing_machine']
#         pred = y_pred_df['washing_machine']
#         score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 5.55
#         test = y_true_df['fridge_freezer']
#         pred = y_pred_df['fridge_freezer']
#         score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 49.79
#         test = y_true_df['TV']
#         pred = y_pred_df['TV']
#         score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 14.57
#         test = y_true_df['kettle']
#         pred = y_pred_df['kettle']
#         score += mt.sqrt(sum((pred.values - test.values) ** 2) / len(test)) * 4.95
#         score /= 74.86
#         return score

```

```python
rnn = RNNModel(60, x_train.shape[2])
rnn.fit(x_train, y_train, x_valid, y_valid, 30)
```

```python
import pandas as pd
val_pred = rnn.predict(x_valid)
y_valid = y_valid.reshape((y_valid.shape[0] * y_valid.shape[1], y_valid.shape[2]))
print(y_valid.shape)
print(val_pred.shape)
rnn.nilm_metric(y_valid, val_pred)
# print(y_train[:10, :])
# print(val_pred[0:10, ])
# print(y_valid[0:10, ])
```
