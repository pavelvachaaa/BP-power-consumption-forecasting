import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import  Flatten, Conv1D,  MaxPooling1D, AveragePooling1D,Activation, AvgPool1D,TimeDistributed,BatchNormalization
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from core import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

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

def add_lags(data_frame: pd.DataFrame, val_name):
    """
    Zpožděné proměnné
    """
    data_frame = data_frame.copy()

    target_map = data_frame[val_name].to_dict()
    data_frame['value_last_year'] = (
        data_frame.index - pd.Timedelta('364 days')).map(target_map)
    # data_frame['value_last_yesterday'] = (data_frame.index - pd.Timedelta('1 days')).map(target_map)
    data_frame['value_last_week'] = (
    data_frame.index - pd.Timedelta('7 days')).map(target_map)
    data_frame['d'] =data_frame[val_name].rolling(window=6).mean()
    data_frame['e'] =data_frame[val_name].rolling(window=12).mean()
    data_frame['g'] =data_frame[val_name].rolling(window=6).std()
    data_frame['i'] =data_frame[val_name].rolling(window=6).max()
    data_frame['l'] =data_frame[val_name].rolling(window=6).min()
    data_frame['energyMean1d'] = (data_frame.index - pd.Timedelta('1 days')).map(target_map)
    data_frame['energyMean7d'] = (data_frame.index - pd.Timedelta('7 days')).map(target_map)
    data_frame['energyMean12h'] = (data_frame.index - pd.Timedelta('12 hours')).map(target_map)
    data_frame['energyMean24h'] = (data_frame.index - pd.Timedelta('24 hours')).map(target_map)
    data_frame['energyMean6h'] = (data_frame.index - pd.Timedelta('6 hours')).map(target_map)
    data_frame['energyMean1h'] = (data_frame.index - pd.Timedelta('1 hours')).map(target_map)
    data_frame["energyMax6h"] = data_frame[val_name].rolling(window = 6).max()

    return data_frame

seq_length = 3

data = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv", "MAC004387", )
data[Y_VALUE_NAME+"_diff"] = data[Y_VALUE_NAME].diff().fillna(0)
data[Y_VALUE_NAME+"_diff2"] = data[Y_VALUE_NAME+"_diff"].diff().fillna(0)
# data = add_lags(data,Y_VALUE_NAME)
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
            Conv1D(filters=64, kernel_size=2, activation='tanh', input_shape=(seq_length, n_features)),
            MaxPooling1D(pool_size=2),
            LSTM(50, activation="tanh", return_sequences=True),
            Dropout(0.15),
            LSTM(75, activation="tanh", return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(1)
            ]
        )
model.compile( optimizer=keras.optimizers.Adam(), metrics=["accuracy"], loss='mse')

lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

# Train the model
"""EarlyStopping(monitor='val_loss', patience=4)"""
 # Define learning rate

history = model.fit(X_train_cnn, y_train,  validation_data=(X_test_cnn, y_test), epochs=25, batch_size=128, verbose=1, callbacks=[lr_scheduler, ],)

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Evaluate the model
mse = model.evaluate(X_test_cnn, y_test, verbose=0)
print(mse)


test_predict = model.predict(X_test_cnn)

# print(test_predict[:,0])
# print("=======")
# print(y_test)
A = y_test[0:512]
F = test_predict[:,0]
F = F[0:512]
evaluate_model(A,F)


size_of_samples = len(F)
aa = [x for x in range(size_of_samples)]

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

# Creating a figure object with desired figure size
plt.figure(figsize=(20, 6))

# Plotting the actual values in blue with a dot marker
plt.plot(df_back.index[0:size_of_samples],A
        [-size_of_samples:], marker='.', label="Naměřená", color='purple')

# Plotting the predicted values in green with a solid line
plt.plot(df_back.index[0:size_of_samples],
        F[-size_of_samples:], '-', label="Predikce", color='red')

# Removing the top spines
sns.despine(top=True)

# Adjusting the subplot location
plt.subplots_adjust(left=0.2)

# Labeling the y-axis
plt.ylabel('kW', size=14)

# Labeling the x-axis
plt.xlabel('Krok', size=14)

# Adding a legend with font size of 15
plt.legend(fontsize=16)

# Display the plot
plt.show()
