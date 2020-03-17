import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from RNN.data_augmenter import DataAugmenter
from RNN.data_imputer import DataImputer
from RNN.rnn_data_formatter import RNNDataFormatter
from RNN.y_imputer import YImputer
from RNN.rnn_model import RNNModel
X_train = pd.read_csv(
    'provided_data_and_metric/X_train_6GWGSxz.csv',
)
Y_train = pd.read_csv(
    'provided_data_and_metric/y_train_2G60rOL.csv',
)
X_test = pd.read_csv(
    'provided_data_and_metric/X_test_c2uBt2s.csv',
)
X_test.drop('Unnamed: 9', axis=1, inplace=True)

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

print('Start of first transform')
x = p1.transform(X_train)
print('End of first transform')
y = p2.transform(Y_train)
print('Second transform')
x_train, y_train = x[:6000, :], y[:6000, :, :]
x_valid, y_valid = x[6000:, :], y[6000:, :, :]

rnn = RNNModel(60)
rnn.fit(x_train, y_train, x_valid, y_valid, 2)


