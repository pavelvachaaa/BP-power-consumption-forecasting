from xgboost import XGBRegressor,plot_importance

import pandas as pd
from core import *

import matplotlib.pyplot as plt
from time import time
import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')


FEATURES = [*WEATHER_DEFAULT_COLUMNS,  *get_time_features_name(), "energyMean1d", "energyMean7d", "energyMean12h", "energyMax6h", "energyMean24h", "energyMean6h"]

if __name__ == "__main__":
    df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_108.csv", "MAC000102", "./data/weather_hourly_darksky.csv", [*WEATHER_DEFAULT_COLUMNS, "precipType"])
    df = add_lags(df)

    df[Y_VALUE_NAME+"_diff"] = df[Y_VALUE_NAME].diff().fillna(0)
    df[Y_VALUE_NAME+"_diff2"] = df[Y_VALUE_NAME+"_diff"].diff().fillna(0)

    unseen_number = int(len(df) * 0.60) # 10 % dat DF neviděl
    df, df_unseen = df[0:unseen_number].copy(), df[unseen_number:len(df)].copy()

    # FEATURES.remove("time")
    # FEATURES = [*FEATURES,  Y_VALUE_NAME+"_diff", Y_VALUE_NAME+"_diff2"]
    FEATURES = ["hour", "energyMax6h", "energyMean1h", "energyMean6h", "energyMean12h", "energyMean7d", Y_VALUE_NAME+"_diff", Y_VALUE_NAME+"_diff2"]

    df = df.sort_index()

    train_size = int(len(df) * 0.70)
    train, test = df[0:train_size], df[train_size:len(df)]

    X_train, y_train = train[FEATURES], train[Y_VALUE_NAME]
    X_test, y_test = test[FEATURES], test[Y_VALUE_NAME]
    df_unseen_X = df_unseen[FEATURES]

    # Validace toho jestli to model náhodou nevidí, protože to je moc úspěšný..
    # df.to_csv("./out/xgboost_validation_df.csv", sep=';', encoding='utf-8')
    # df_unseen_X.to_csv("./out/xgboost_df_unseen.csv", sep=';', encoding='utf-8')
    
    reg = XGBRegressor( n_estimators=1000,
                       base_score=0.5, booster='gbtree',
                       objective='reg:squarederror',
                       max_depth=6,
                       early_stopping_rounds=100,
                       eta=0.01,
                       learning_rate=0.02
    )
    
    reg.fit(
            X_train, y_train,
            eval_metric="logloss",
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100,
    )

    evals_result = reg.evals_result()


    ax = plot_importance(reg, height=0.9,
                    xlabel='Skóre příznaku', ylabel='Příznak', title="", values_format= "{v:.0f}",xlim=(0,14500), max_num_features=10) 

    plt.savefig('./out/xgboost_f_score.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()

    df_unseen['pred'] = reg.predict(df_unseen_X)

  
    import random
    time_window = 48 *1
    ax = df_unseen[[Y_VALUE_NAME, 'pred']][:time_window].plot(figsize=(20, 6),xlabel="Čas", title="",linewidth=2, ylabel="Spotřeba energie [kW/h]")
    plt.savefig(f'./out/apendix/xgboost/prediction_{random.randint(0,99999)}.eps', format='eps', bbox_inches='tight', transparent=True, )

    evaluate_model(df_unseen[Y_VALUE_NAME][:time_window],df_unseen["pred"][:time_window])
    serialize_model(reg, "xgboost", "MAC000291")


    train_rmse = evals_result['validation_0']['logloss']
    val_rmse = evals_result['validation_1']['logloss']

    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse, label='Trénovací')
    plt.plot(val_rmse, label='Validační')
    plt.xlabel('Počet iterací')
    plt.ylabel('Ztráta')
    plt.legend()
    plt.grid(True)
    plt.savefig('./out/xgboost_learning_curve.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()