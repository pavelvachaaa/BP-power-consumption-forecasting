import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core import *
import matplotlib.pyplot as plt

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')
from statsmodels.tsa.arima.model import ARIMAResults, ARIMAResultsWrapper
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

from sklearn.metrics import roc_curve, auc,recall_score,precision_score

import statsmodels.api as sm



if __name__ == "__main__":
    df: pd.DataFrame = load_agg_dataseet("./data/agg_halfhourly.csv")
    df = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_7.csv", "MAC004385", )

    df = df.resample('1H').mean().astype('float32')

    model = ARIMAResults.load("./out/models/arima_arma_(0, 0, 1)_ses_(1, 0, 1, 24).pkl")
 

    
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    predictions = model.forecast(steps=len(test))

    evaluate_model(test[Y_VALUE_NAME], predictions)

    plt.figure(figsize=(12, 6))
    # plt.plot(train.index[:96*5], train[Y_VALUE_NAME][:96*5], label='Training')
    plt.plot(test.index[:96*5], test[Y_VALUE_NAME][:96*5], label='Test')
    plt.plot(test.index[:96*5], predictions[:96*5], label='Predictions')
    plt.title('SARIMA Forecast')
    plt.legend()
    plt.show()