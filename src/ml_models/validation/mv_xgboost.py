import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core import *
import matplotlib.pyplot as plt

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

from sklearn.metrics import roc_curve, auc,recall_score,precision_score

# Ještě zkusit překopat na oříznutí DF a pak použít predict_future a porovnat to s tím (mělo by to vyjít stejně)
if __name__ == "__main__":
    # Zde je použit kompletně jiná část londýna a náhodný barák pro validaci modelu, který byl natrénován také na úplně jiné části londýna a jiném baráku
    df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_12.csv", "MAC000291", "./data/weather_hourly_darksky.csv", [*WEATHER_DEFAULT_COLUMNS, "precipType"])
    df = df[:15*4*2*10]
    # Features, které to potřebuje
    df = add_lags(df)
    df[Y_VALUE_NAME+"_diff"] = df[Y_VALUE_NAME].diff().fillna(0)
    df[Y_VALUE_NAME+"_diff2"] = df[Y_VALUE_NAME+"_diff"].diff().fillna(0)
    
    FEATURES = ["hour", "energyMax6h", "energyMean1h", "energyMean6h", "energyMean12h", "energyMean7d", Y_VALUE_NAME+"_diff", Y_VALUE_NAME+"_diff2"]
    df = df.sort_index()

    xgb_model_loaded = deserialize_model("xgboost","beast_0_MAC004431_ULTRA")

    X_vals, Y_vals = df[FEATURES], df[Y_VALUE_NAME]
 
    df['pred'] = xgb_model_loaded.predict(X_vals)

    _ = df[[Y_VALUE_NAME, 'pred']].plot(figsize=(20, 6))
    
    actual_values = Y_vals
    forecast_values = df["pred"]
    
    evaluate_model(actual_values,forecast_values)

    plt.show()

