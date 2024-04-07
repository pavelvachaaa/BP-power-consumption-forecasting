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


import logging

def fit_sarima(data, order, seasonal_order):
    model = sm.tsa.arima.ARIMA(data[Y_VALUE_NAME], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()

    return results

if __name__ == "__main__":
    logger = logging.getLogger("finder")
    logging.basicConfig(filename='./out/arima_finder.log', encoding='utf-8', level=logging.INFO)

    df: pd.DataFrame = load_agg_dataseet("./data/agg_halfhourly.csv")
    df = df.resample('1h').mean().astype('float32')
 
    order = (0,0, 1)
    seasonal_order = (1, 0, 1, 24)  

    train_size = int(len(df) * 0.9)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    logger.info("STARTING ANALYSIS")

    best_aic = float("inf")
    best_model = None
    best_params = None

    for p in [2, 3]:
        for q in [2, 3]:
            for P in [1]:
                for Q in [1]:
                    current_order = (p, 0, q)
                    current_seasonal_order = (P, 0, Q, 24)

                    sarima_model = fit_sarima(train, current_order, current_seasonal_order)
                    predictions = sarima_model.forecast(steps=len(test))
                    current_aic = sarima_model.aic
                    current_mse = sarima_model.mse
                    current_sse = sarima_model.sse

                    evaluate_model(test[Y_VALUE_NAME], predictions)

                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_model = sarima_model
                        best_params = (current_order, current_seasonal_order)

                    logger.info(f"Order: {current_order}, Seasonal Order: {current_seasonal_order}, AIC: {current_aic} AAIC {abs(current_aic)} mse: {current_mse} sse: {current_sse} ")
                    print(f"Order: {current_order}, Seasonal Order: {current_seasonal_order}, AIC: {current_aic}  AAIC {abs(current_aic)}  mse: {current_mse} sse: {current_sse} ")

    logger.info(f"Best Model - Order: {best_params[0]}, Seasonal Order: {best_params[1]}, AIC: {best_aic}")
    print(f"Best Model - Order: {best_params[0]}, Seasonal Order: {best_params[1]}, AIC: {best_aic}")

