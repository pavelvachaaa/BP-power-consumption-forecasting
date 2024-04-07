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


def fit_sarima(data, order, seasonal_order,id: str):
    model = sm.tsa.arima.ARIMA(data[Y_VALUE_NAME], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    # results.save(f'./out/models/arima_arma_{order}_ses_{seasonal_order}_{id}.pkl')

    return results

if __name__ == "__main__":
    household = "MAC000291"
    df = load_london_dataset_household(f"./data/halfhourly_dataset/halfhourly_dataset/block_12.csv", household, )
    # df: pd.DataFrame = load_agg_dataseet("./data/agg_halfhourly.csv")
    # df: pd.DataFrame = load_iris_dataset("./data/albistech_dataset/db3.json")

    df = df.resample('1h').mean().astype('float32')
    order = (0,0,1)
    seasonal_order = (1,0, 1, 24)

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    sarima_model = fit_sarima(train, order, seasonal_order,household)

    predictions = sarima_model.forecast(steps=len(test))

    # print(f"Block: {block}, LCLid: {lclid}")
    print("=================================")

    print("24h")
    print("=======")
    evaluate_model(test[Y_VALUE_NAME][:24], predictions[:24])
    print("\n")
    print("\n")
    print("7d")
    print("========")
    evaluate_model(test[Y_VALUE_NAME][:24*7], predictions[:24*7])
    print("\n")
    print("\n")

    plt.figure(figsize=(12, 6))
    plt.xlabel("Čas")
    plt.ylabel("Spotřeba energie [kW/h]")
    plt.title("")
    # plt.plot(train.index[:96*5], train[Y_VALUE_NAME][:96*5], label='Training')
    plt.plot(test.index[:24*5], test[Y_VALUE_NAME][:24*5], label='Naměřená', linewidth=2,)
    plt.plot(test.index[:24*5], predictions[:24*5], label='Predikce', linewidth=2,)
   
    plt.legend()
    plt.savefig('./out/spike.eps', format='eps', bbox_inches='tight', transparent=True)

    plt.show()