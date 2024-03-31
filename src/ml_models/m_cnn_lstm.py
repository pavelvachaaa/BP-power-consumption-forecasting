import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import  Flatten, Conv1D,  MaxPooling1D, AveragePooling1D,Activation, AvgPool1D,TimeDistributed,BatchNormalization, RepeatVector
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from core import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

from keras.optimizers import Adam

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
    seq_length = 48

    data = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv", "MAC004431", )
    data[Y_VALUE_NAME+"_diff"] = data[Y_VALUE_NAME].diff().fillna(0)
    data[Y_VALUE_NAME+"_diff2"] = data[Y_VALUE_NAME+"_diff"].diff().fillna(0)

    df_back = data
    n_features = data.shape[1]

    X, y, scaler = preprocess_data(data, seq_length)

    split = int(0.80 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

    model = keras.Sequential(
                [
                Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(seq_length, n_features)),
                # Conv1D(filters=32, kernel_size=1, ),
                MaxPooling1D(pool_size=2),
                LSTM(75, activation="tanh", return_sequences=True),
                Dropout(0.15),
                LSTM(100, activation="tanh", return_sequences=True),
                Dropout(0.15),
                LSTM(75, activation="tanh", return_sequences=False),
                Dropout(0.15),
                Dense(1)
                ]
            )
    model.compile( optimizer=keras.optimizers.Adam(), metrics=["accuracy"], loss='mse')
    model.summary()
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

    history = model.fit(X_train_cnn, y_train,  validation_data=(X_test_cnn, y_test), epochs=20, batch_size=128, verbose=1, callbacks=[lr_scheduler, ],)


    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('')
    plt.ylabel('Ztráta')
    plt.xlabel('Epocha')
    plt.legend(['Trénovací', 'Validační'], loc='upper right')
    plt.grid(True)
    plt.savefig('./out/cnn_lstm_learning_curve.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()
    # Evaluate the model
    mse = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(mse)

    serialize_model(model, "cnn_lstm","beast2")
    loaded_model = deserialize_model("cnn_lstm", "beast2")
    test_predict = loaded_model.predict(X_test_cnn)


    # BIG TODO: Inverse scale, jinak budou kWh v mínusu :DDD

    # print(test_predict[:,0])
    # print("=======")
    # print(y_test)
    A = y_test[0:48]
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
    plt.savefig('./out/cnn_lstm_vyrez.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()
