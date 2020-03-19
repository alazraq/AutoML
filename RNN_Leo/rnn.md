```python
import math as mt
import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, SimpleRNN, TimeDistributed
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
X_test.drop('Unnamed: 9', axis=1, inplace=True)
X_test.set_index("time_step", inplace=True)

px = XPipeline()
py = YPipeline()
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
x_train, y_train = x[:6000, :], y[:6000, :, :]
x_valid, y_valid = x[6000:, :], y[6000:, :, :]
```

```python
print(f"x_train shape is {x_train.shape}")
print(f"x_valid shape is {x_valid.shape}")
```

```python
class RNNModel:

    COLUMNS = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']

    def __init__(self, batch_size=60, features=57):
        self.history = None
        self.batch_size = batch_size
        self.features = features
        self.model = Sequential()
        
        self.model.add(
            LSTM(
                units=64,
                dropout=0.4,
                input_shape=[None, self.features],
                return_sequences=True
            )
        )
        self.model.add(
            LSTM(
                units=32,
                dropout=0.5,
                return_sequences=True,
#                 input_shape=[None, self.features]
            )
        )
        self.model.add(
            Dense(
                units=4
            )
        )
        self.model.compile(
            loss="mse", 
            optimizer=Adam(lr=0.001), 
        )
        

    def fit(self, x_train, y_train, x_valid, y_valid, epochs=10):
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=(x_valid, y_valid)
        )

    def predict(self, x_test):
        pred = self.model.predict(x_test)
        pred = pred.reshape((pred.shape[0] * pred.shape[1], pred.shape[2]))
        return pd.DataFrame(pred, columns=RNNModel.COLUMNS)

    def nilm_metric(self, y_true, y_pred):
        if not isinstance(y_true, pd.DataFrame):
            y_true = y_true.reshape((y_true.shape[0] * y_true.shape[1], y_true.shape[2]))
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
```

```python
rnn = RNNModel(60)
rnn.fit(x_train, y_train, x_valid, y_valid, 20)
```

```python
pred = rnn.predict(x_valid)
```

```python
(Y_train.washing_machine > 100).sum()
```

```python
pred
```

```python
(pred.washing_machine > 6).sum()
```

```python
rnn.nilm_metric(y_valid, pred)
```
