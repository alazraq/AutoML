from keras.layers import LSTM, Dense, Flatten, Dropout, Activation, SimpleRNN
from keras.models import Sequential
import pandas as pd
import math as mt


class RNNModel:

    COLUMNS = ['washing_machine', 'fridge_freezer', 'TV', 'kettle']

    def __init__(self, batch_size=60, features=57):
        self.history = None
        self.batch_size = batch_size
        self.features = features
        self.model = Sequential()
        
        self.model.add(
            SimpleRNN(
                units=20,
                input_shape=[None, self.features],
                return_sequences=True
            )
        )
        self.model.add(
            LSTM(
                units=20,
                return_sequences=True,
#                 input_shape=[None, self.features]
            )
        )
        self.model.add(
            LSTM(
                19,
                return_sequences=True
            )
        )
#         self.model.add(
#             LSTM(
#                 19
#             )
#         )
        self.model.add(
            Dense(
                units=4
            )
        )
        self.model.compile(
            loss="mse", 
            optimizer="adam", 
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
