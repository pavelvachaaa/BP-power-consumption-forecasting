import pandas as pd
from core import *
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

is_albistech = False


def fit_sarima(data, order, seasonal_order):
    model = sm.tsa.statespace.SARIMAX(data[Y_VALUE_NAME], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    return results

def create_lag_features(data, lag_features, lag_periods):
    for feature in lag_features:
        for period in lag_periods:
            data[f'{feature}_lag_{period}'] = data[feature].shift(period)
    return data


if __name__ == "__main__":
    df: pd.DataFrame = load_agg_dataseet("./data/agg_halfhourly.csv")
    order = (1,1, 1)
    seasonal_order = (1, 1, 1, 48)  
    lag_features = ['energy(kWh/hh)']  # Features to create lagged versions of
    lag_periods = [1, 2, 24]  # Lag periods to consider

    df = create_lag_features(df, lag_features, lag_periods)


    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    sarima_model = fit_sarima(train, order, seasonal_order)

    predictions = sarima_model.forecast(steps=len(test))

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(test[Y_VALUE_NAME], predictions)
    rmse = mean_squared_error(test[Y_VALUE_NAME], predictions, squared=False)
    mape = mean_absolute_percentage_error(test[Y_VALUE_NAME][:96*5], predictions[:96*5])

    print(mae)
    print(rmse)
    print(mape)

    plt.figure(figsize=(12, 6))
    # plt.plot(train.index[:96*5], train[Y_VALUE_NAME][:96*5], label='Training')
    plt.plot(test.index[:96*5], test[Y_VALUE_NAME][:96*5], label='Test')
    plt.plot(test.index[:96*5], predictions[:96*5], label='Predictions')
    plt.title('SARIMA Forecast')
    plt.legend()
    plt.show()