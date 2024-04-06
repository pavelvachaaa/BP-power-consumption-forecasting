import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core import *
from NNModels import *

import matplotlib.pyplot as plt
from keras.models import load_model
import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

import tensorflow as tf




def preprocess_data(df, seq_length):
    # Normalizace
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # lookback 
    sequences = []
    labels = []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data[i:i+seq_length])
        labels.append(scaled_data[i+seq_length, 0])  # Energie je první sloupec
    return np.array(sequences), np.array(labels), scaler


if __name__=="__main__":
    is_albistech = False

    seq_length = 48

    df = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_12.csv", "MAC000291", )
    df[Y_VALUE_NAME+"_diff"] = df[Y_VALUE_NAME].diff().fillna(0)
    df[Y_VALUE_NAME+"_diff2"] = df[Y_VALUE_NAME+"_diff"].diff().fillna(0)

    ONE_DAY = 96 if is_albistech else 48
    DAYS_OF_PREDICTION =  ONE_DAY*5

    n_features = df.shape[1]
    df= df.iloc[-DAYS_OF_PREDICTION:].copy()
    df_back = df

    # Provedeme transformaci hodnot dataframu
    X, y, scaler = preprocess_data(df, seq_length)

    X_train_cnn = X.reshape((X.shape[0], X.shape[1], n_features))
    
    model = deserialize_model("cnn_lstm", "beast2")
    test_predict = model.predict(X_train_cnn)

    # BIG TODO: Inverse scale, jinak budou kWh v mínusu :DDD


    A = y[0:48]
    F = test_predict[:,0]
    F = F[0:48]
    evaluate_model(A,F)



    size_of_samples = len(F)
    aa = [x for x in range(size_of_samples)]

    plt.figure(figsize=(20, 6))
    plt.plot(df_back.index[0:size_of_samples],A
            [-size_of_samples:], marker='.', label="Naměřená", color='purple')

    plt.plot(df_back.index[0:size_of_samples],
            F[-size_of_samples:], '-', label="Predikce", color='red')
    sns.despine(top=True)
    plt.subplots_adjust(left=0.2)
    plt.ylabel('kW', size=14)
    plt.xlabel('Krok', size=14)
    plt.legend(fontsize=16)

    plt.grid(True)
    # plt.savefig('./out/cnn_lstm_vyrez.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()