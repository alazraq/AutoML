```python
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

from x_pipeline import XPipeline
from y_pipeline import YPipeline
```

```python
X_train = pd.read_csv(
    '../../provided_data_and_metric/X_train_6GWGSxz.csv',
)
X_train.set_index("time_step", inplace=True)
Y_train = pd.read_csv(
    '../../provided_data_and_metric/y_train_2G60rOL.csv',
)
Y_train.set_index("time_step", inplace=True)
X_test = pd.read_csv(
    '../../provided_data_and_metric/X_test_c2uBt2s.csv',
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
print('Start of first transform')
X = px.fit(X_train)
X = px.transform(X_train)
print('End of first transform')
y = py.fit(Y_train)
y = py.transform(Y_train)
print('Second transform')
```

```python
hour_mean = y.groupby(X.hour).mean()
hour_mean.head()
```

```python
for i in hour_mean.index:
    X.loc[X.hour == i, "hour_mean"] = float(hour_mean.loc[i])
```

```python
X.head()
```

## Exploration

```python
y.loc['2013-12-31 17:11:00':'2013-12-31 17:17:00']
```

```python
X.loc['2013-12-31 17:11:00':'2013-12-31 17:17:00', "consumption"]
```

## Modeling

```python
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=49)
```

```python
print(f"x_train shape is {x_train.shape}")
print(f"x_valid shape is {x_val.shape}")
```

```python
def nilm_metric(y_true, y_pred):
        score = 0.0
        score += math.sqrt(sum((y_pred.get_label() - y_true) ** 2) / len(y_true)) * 4.95
        score /= 74.86
        return "nilm", score
```

```python
xgb_reg = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=500, random_state=42)

xgb_reg.fit(x_train, y_train,
            eval_set=[(x_val, y_val)],
            eval_metric=nilm_metric,
#             early_stopping_rounds=10
           )
```

```python
y_pred = xgb_reg.predict(x_val)   
```

```python
pred_big = y_pred[y_pred>500]
true_big = y_val.kettle[y_pred>500]

ax = sns.scatterplot(x=true_big, y=pred_big)
ax.set(xlabel='true', ylabel='predicted')
plt.show()
```

```python
ax = sns.scatterplot(x=y_val.kettle, y=y_pred)
ax.set(xlabel='true', ylabel='predicted')
plt.show()
```

```python
importances = xgb_reg.feature_importances_
#std = np.std([tree.feature_importances_ for tree in name.estimators_], axis=0)
# add yerr = std yadda yadda in plt bar in case you need this
indices = np.argsort(importances)[::-1]
indices = indices[:15]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(indices)):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the name
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), indices)
plt.xlim([-1, len(indices)])
plt.show()
```

### Evaluating Performance

```python
pred = pd.DataFrame(y_pred, columns=["kettle"])
```

```python
def nilm_metric(y_true, y_pred):
        score = 0.0
        score += math.sqrt(sum((y_pred.values - y_true.values) ** 2) / len(y_true)) * 4.95
        score /= 74.86
        return score
```

```python
nilm_metric(y_val, pred)
```

```python
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(pred.kettle, y_val.kettle)
plt.plot(np.linspace(0,3000), np.linspace(0,3000), c="orange")
plt.show()
```

### submission

```python
X_test = pd.read_csv(
    '../../provided_data_and_metric/X_test_c2uBt2s.csv',
)
time = X_test["time_step"]
X_test.set_index("time_step", inplace=True)
```

```python
x_test = px.transform(X_test)
```

```python
for i in hour_mean.index:
    x_test.loc[x_test.hour == i, "hour_mean"] = float(hour_mean.loc[i])
```

```python
pred = xgb_reg.predict(x_test)
pred = pd.DataFrame(pred, columns=["kettle"])
```

```python
pred = pd.concat([time, pred], axis=1)
```

```python
pred.to_csv("kettle.csv", index=False)
```
