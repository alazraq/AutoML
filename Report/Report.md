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

# Final Report for Machine Learning II Data Challenge: Smart meter is coming
by Guillaume Le Fur, Abderrahmane Lazraq and Leonardo Natale


## Outline
1. Introduction: objectives and methodology
2. Exploratory Data Analysis
3. Data preprocessing 
4. Feature engineering by appliance
    - Washing machine
    - Fridge/ Freezer
    - TV
    - Kettle
4. First approach: deep learning
5. Second approach: ensemble methods - Boosting
6. Results and benchmark
7. Conclusion


## Introduction: objectives and methodology


The objective of this project is to put the Machine Learning methods that we've been taugh during this course into practice, on a real data set, the "Smart meter is coming" challenge.

We will start by introducing our exploratory data analysis and what first conclusions we could draw from it. Then, we'll detail the data pre-processing and feature engineering we've done, and ustify their interest.

Finally, we'll present the results we had using two methods : Deep learning (with RNNs) and Boosting (with XGboost).

You will be able to find the entirety of the code on the following [GitHub repository](https://github.com/alazraq/AutoML). Not all the code will be detailed here but rather the most important parts.

# <font color='red'>TODO data description</font>


## Exploratory Data Analysis


### Global vs. per appliance consumption


First of all, if we denote by $\mathcal A$ the ensemble of appliances, $c_a$ the consumption of appliance $a \in \mathcal A$ and $c_{tot}$ the total consumption, it is important to emphasize the fact that, for each timestamp, we don't have : 

$$\sum_{a \in A} c_a = c_{tot}$$

We can clearly see this on the following plot.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
X_train = pd.read_csv(
    '../provided_data_and_metric/X_train_6GWGSxz.csv',
)
X_train.set_index("time_step", inplace=True)
X_train.index = pd.to_datetime(X_train.index)
Y_train = pd.read_csv(
    '../provided_data_and_metric/y_train_2G60rOL.csv',
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
ax.set_xlabel("Time")
ax.set_ylabel("Consumption")
ax.set_title("Weekly average of every appliance and total consumption.")
f.canvas.draw()
```

But this plot is not precise enough. Instead, if we look at the daily moving average over 7 days, we have :

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
ax.set_xlabel("Time")
ax.set_ylabel("Consumption")
ax.set_title("Daily moving average over 7 days of every appliance and total consumption.")
ax.legend(loc='upper left')
f.canvas.draw()
```

On the graph above, we can clearly see that the overall consumption trend does not correspond to any per-appliance trend. Indeed, we can observe two sharp declines (one around 2013-04-15, and another around 2013-08-10) that lead to an opposite effect on the per-appliance trends (on the first one, the per-appliance average drops, and on the second it raises). This makes it even harder to predict the per-appliance consumption as there is no clear link between them and the overall consumption.

The difference between the consumtions can most probably be explained by the presence of other appliances in the house.

This means that predicting the value of the appliance and it's percentage of the consumption is not the same problem.


Now let's have a look at some specificities of the data.


### Analysis of the predictors

```python
import holidays

def add_features(x):
    x = x.drop(
        ['Unnamed: 9', 'visibility', 'humidity', 'humidex', 'windchill', 'wind', 'pressure', 'temperature'],
        axis=1
    )
    fr_holidays = holidays.France()
    x["weekday"] = x.index.dayofweek
    x["month"] = x.index.month
    x["hour"] = x.index.hour
    x["is_weekend"] = (x["weekday"] > 4) * 1
    x["is_holidays"] = (x.index.to_series().apply(lambda t: t in fr_holidays)) * 1

    x["is_breakfast"] = ((x.hour > 5) & (x.hour < 9)) * 1
    x["is_teatime"] = ((x.hour > 16) & (x.hour < 20)) * 1
    x["is_TVtime"] = ((x.hour > 17) & (x.hour < 23)) * 1
    # X_train["is_working_hour"] = ((X_train.hour>7) & (X_train.hour<19))*1
    x["is_night"] = ((x.hour > 0) & (x.hour < 7)) * 1
    return x
X_data_exploration = add_features(X_train)
```

```python
X_data_exploration[["consumption", "is_weekend"]].groupby("is_weekend").mean()
```

The overall consumption is **higher during the weekend**, as expected.

```python
X_data_exploration[["consumption", "weekday"]].groupby("weekday").mean().plot()
```

The consumption is also really **high on tuesday**. We couldn't find any justification for this

```python
X_data_exploration[["consumption", "month"]].groupby("month").mean().plot()
```

The consumption is **higher during *cold months*** (October to February). This might be due to the **heating system** which works more in winter than in summer.

```python
X_data_exploration[["consumption", "hour"]].groupby("hour").mean().plot()
```

The hourly consumption is quite interesting. Indeed, we can see that most of the consumption is done **after 4 p.m.**, which is after the end of *office hours*, when people are back home, and **before 11 p.m.**, when people go to sleep. There are also two smaller *peaks*, suring **breakfast** and **luch time**.

```python
X_data_exploration[["consumption", "is_holidays"]].groupby("is_holidays").mean()
```

The consumption is **higher during the holidays**. Our analysis led us to believe that the data was coming from a **house located in France** because the data was fitting better the holidays in France than the ones in the UK or in the US.


### Analysis of the response variables

```python
Y_train.groupby(X_data_exploration.weekday).mean()
```

We can see that people tend to use their **washing machine more on Sundays**, which is logical because they have more time on Sundays and **electricity is cheaper**. If we consider, as we did, that the houseis in France, people most likely beneficiate from the *Heures Creuses* rate.

```python
Y_data_exploration.groupby(X_data_exploration.month).mean()
```

```python
Y_data_exploration.groupby(X_data_exploration.month).mean().plot()
```

We detect a significant increase of the use of the **Kettle in November**, which also makes sense because it's one of the first 'cold' months so people strat making tea again to warm themselves.

```python
Y_train.groupby(X_data_exploration.is_weekend).mean()
```

Once again, the use of the **washing machine on the weekend** is confirmed here. People tend to use their **kettle a bit more** as well. **We could have expected the consumption of the TV to be higher** on the weekend but it actually isn't.

```python
Y_train.groupby(X_data_exploration.hour).mean().plot()
```

From the plot above, we can extract the following informations:
    
- People use their **TV in the morning**, really early, **and in the evening**, but not much after 11 p.m., after the main movie has finished.
- People use their **kettle around teatime**, which is quite logical, but also a bit in the morining, **for breakfast**.
- The consumption fo the **freezer doesn't vary much** during the day.
- People tend to turn their **washing machine on when they go to bed**, once again for the **cost of electricity**.

```python
Y_train.groupby(X_data_exploration.is_holidays).mean()
```

**People don't use their wshing machine on holidays, nor their kettle**. This makes sense becasue when people leave the house, thei appliances that consume a lot of electricity when used aren't used any longer so they stop consuming, while the appliances that consume an almost constant amount of electricity don't vary much because they keep working.


For all these reasons, we thought it would be relevant to **add some features to the data**, to be able to predict the per-appliance consumption with more accuracy. This will be detailed further in this report.


# <font color='red'>TODO add other plots</font>


## Data preprocessing


We define two pipelines for the input dataset, one for each ML approach we attempted to make data compatible with it. 


### 1. Pipeline adapted to XGBoost

```python
class XPipeline_XGB:
    def __init__(self):
        self.pipeline = Pipeline([
            ('DataImputer', DataImputer()),
            ('MyStandardScaler', MyStandardScaler()),
            ('DataAugmenter', DataAugmenter()),
        ])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)
```

There are three steps in our pipeline:


- A DataImputer that drops the unuseful columns, interpolates missing values linearly and sets the date as the index

```python
class DataImputer(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.X = None

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        try:
            x.drop(['Unnamed: 9', 'visibility', 'humidity', 'humidex', 'windchill', 'wind', 'pressure'],
                   axis=1, 
                   inplace=True)
        except KeyError as e:
            pass
        x = x.interpolate(method='linear').fillna(method='bfill')
        x.index = pd.to_datetime(x.index)
        return x
```

- A standard scaler that standardizes features by removing the mean and scaling to unit variance

```python
class MyStandardScaler(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        self.index = X.index
        X = pd.DataFrame(self.scaler.transform(X),
                         columns=self.columns, 
                         index=self.index
                        )
        return X
```

- A Data Augmenter for feature engineering. There is a different data augmenter for each appliance, we inspect those in detail in the following section.


### 2. Pipeline adapted to RNN

```python
class XPipeline_RNN:

    def __init__(self):
        self.pipeline = Pipeline([
            ('DataImputer', DataImputer()),
            ('MyStandardScaler', MyStandardScaler()),
            ('DataAugmenter', DataAugmenter()),
            ('MyOneHotEncoder', MyOneHotEncoder()),
            ('RNNDataFormatter', RNNDataFormatter())
    ])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)
```

We only add two additional steps to make data compatible with RNN:


- A One Hot Encoder for the categorical features (hours, weekdays and months)

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
        hours = pd.DataFrame(self.ohe_hours.transform(X.hour.values.reshape(-1,1)).toarray(), 
                             columns=["hour_"+str(i) for i in range(1, 24)],
                             index=X.index
                            )
        weekdays = pd.DataFrame(self.ohe_weekdays.transform(X.weekday.values.reshape(-1,1)).toarray(),
                                columns=["weekday_"+str(i) for i in range(1, 7)],
                                index=X.index
                               )
        months = pd.DataFrame(self.ohe_months.transform(X.month.values.reshape(-1,1)).toarray(), 
                              columns=["month_"+str(i) for i in range(2, 13)],
                              index=X.index
                             )
        X = pd.concat([X, hours, weekdays, months], axis=1)
        X.drop(["month", "weekday", "hour"], axis=1, inplace=True)
        return X
```

- A data formatter to produce batches dor RNN, 60 observations per batch (1 hour of observations) seemed like a reasonable choice

```python
class RNNDataFormatter(BaseEstimator, TransformerMixin):
    
    def __init__(self, batch_size=60):
        self.X = None
        self.batch_size = batch_size
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        print(x.shape)
        print(x.__class__.__name__)
        while x.shape[0] % self.batch_size != 0:
            print("Appending a row")
            print([x[-1, :]])
            x = np.append(x, [x[-1, :]], axis=0)
        print(x.shape)
        nb_col = x.shape[1]
        return x.reshape((int(x.shape[0] / self.batch_size), self.batch_size, nb_col))
```

##  Feature engineering by appliance


For each appliance we produced additional features that aim at increasing the predictive power of the machine learning algorithms used by creating features from the raw data that help facilitate the machine learning process for that specific appliance. 

The most important features that we identified to transform the time series forecasting problem into a supervised learning problem are the lag features and the rolling mean. Here we focus on the different lags and rolling means used for each appliance, as well as other features specific to each appliance.


### 1. Washing machine


### 2. Fridge/ Freezer


### 3. TV


### 4. Kettle


# <font color='red'>TODO Should we add MultiOutputRegressor + Random Forests as our baseline?</font>


## First approach: deep learning

```python

```

## Second approach: ensemble methods - Boosting

```python

```

## Results and benchmark

```python

```

## Conclusion

```python

```

```python
######################################### Copy of Leo.md for future use #############################################
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

