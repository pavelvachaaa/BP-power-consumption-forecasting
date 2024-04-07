import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import logging

is_albistech = False


def fit_sarima(data, order, seasonal_order):
    model = sm.tsa.arima.ARIMA(data[Y_VALUE_NAME], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()


    return results

if __name__ == "__main__":
    logger = logging.getLogger("arima_evaluator")
    logging.basicConfig(filename='./out/arima_evaluator.log', encoding='utf-8', level=logging.INFO)
    logger.info("New iter")
    blocks = [12, 0, 7]
    lclids = ["MAC000291", "MAC003597", "MAC004385"]

    for block, lclid in zip(blocks, lclids):
        df = load_london_dataset_household(f"./data/halfhourly_dataset/halfhourly_dataset/block_{block}.csv", lclid, )
        df = df.resample('1h').mean().astype('float32')
        order = (1,0,12)
        seasonal_order = (1, 0, 1, 24)

        train_size = int(len(df) * 0.8)
        train, test = df.iloc[:train_size], df.iloc[train_size:]

        sarima_model = fit_sarima(train, order, seasonal_order)

        predictions = sarima_model.forecast(steps=len(test))

        logger.info(f"Block: {block}, LCLid: {lclid}")
        logger.info("=================================")
        logger.info("24h")
        logger.info("=======")
        evaluate_model(test[Y_VALUE_NAME][:24], predictions[:24], logger)
        logger.info("\n")
        logger.info("\n")
        logger.info("7d")
        logger.info("========")
        evaluate_model(test[Y_VALUE_NAME][:24*7], predictions[:24*7], logger)
        logger.info("\n")
        logger.info("\n")

   