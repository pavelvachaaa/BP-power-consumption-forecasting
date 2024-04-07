import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.layers import  Flatten, Conv1D, BatchNormalization, MaxPooling1D, AveragePooling1D,Activation, AvgPool1D,TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from keras import metrics

import keras

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
    mean_squared_error

class NNModels:
    """
    Wrapper okolo LSTM a CNN LSTM
    """
    scaler: MinMaxScaler    

    def __init__(self, scaler):
        self.scaler = scaler

    def reshape_for_lstm(self, x_train: np.ndarray) -> np.ndarray:
        """
        reshape input to be [samples, time steps, features]
        """
        return (np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])))
    
    def to_sequence_for_lstm(self, data: np.ndarray[any, any], look_back=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts an array of values into a dataset matrix
        """
        X, Y = [], []
        for i in range(len(data)-look_back-1):
            a_temp = data[i:(i+look_back), 0]
            X.append(a_temp)
            Y.append(data[i + look_back, 0])
            
        return np.array(X), np.array(Y) 
    
    def transform_for_lstm(self, data: np.ndarray[any]) -> np.ndarray:
        """
        Normalizace na -1 až 1
        """
        data = np.reshape(data, (-1,1))
        return self.scaler.fit_transform(data)
    
    def add_lags(self,data_frame: pd.DataFrame, val_name):
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


    def split_dataset(self, data, train_size=0.70) -> tuple[np.ndarray, np.ndarray]:
        """
        Rozdělí datovou sadu na testovací a trénovací. 
        Data se nemíchají
        """
        train_size = int(len(data) * train_size)
        return (data[0:train_size,
                     :], data[train_size:len(data), :])
    

    def model_lstm_one(self, train_input_shape: tuple) -> Sequential:
        """
        Testovací model LSTM
        """
        model = keras.Sequential(
            [
            # RepeatVector(50),
            LSTM(75, activation="tanh", return_sequences=True, input_shape=(train_input_shape[1], train_input_shape[2])),
            Dropout(0.15),
            LSTM(100, activation="tanh", return_sequences=True),
            Dropout(0.15),
            LSTM(75, activation="tanh", return_sequences=False),
            Dropout(0.15),
            Dense(1),
            ]
        )
        
        model.summary()
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

        
        return model
    
    def model_cnn_lstm(self, train_input_shape:tuple) -> Sequential:
        model = keras.Sequential(
            [
                
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(64,64)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            LSTM(50, activation="tanh", return_sequences=True),
            Dropout(0.15),
            LSTM(75, activation="tanh", return_sequences=True),
            Dropout(0.15),
            LSTM(50, activation="tanh", return_sequences=False),
            Dropout(0.15),
            Dense(1)
            ]
        )
        
        model.summary()
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

        
        return model
    
    def model_save(self, model: Sequential, code_name: str) -> str:
        """
        Vrací název uloženého souboru
        """
        filename = f'./out/models/lstm_model_{code_name}.h5'
        model.save(filename)
        return filename

    def get_model(self, code_name: str):
        """
        Vrací model na základě zadané cesty
        """
        return keras.models.load_model(f"./out/models/lstm_model_{code_name}.h5")
