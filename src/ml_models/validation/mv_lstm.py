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
is_albistech = False


wrapper = NNModels(scaler=MinMaxScaler(feature_range=(0, 1)))
df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_12.csv", "MAC000291")

ONE_DAY = 96 if is_albistech else 48
DAYS_OF_PREDICTION =  ONE_DAY*1

# Extrahujeme časové features
df = wrapper.add_lags(df, Y_VALUE_NAME)

df= df.iloc[-DAYS_OF_PREDICTION:].copy()
df_back = df

# Ujistéme se, že naše Y_VAL je float64
df = df[Y_VALUE_NAME].values.astype('float64')
df[Y_VALUE_NAME+"_diff"] = df[Y_VALUE_NAME].diff().fillna(0)

# Provedeme transformaci hodnot dataframu
dataset = wrapper.transform_for_lstm(df)

test_data, test_y = wrapper.to_sequence_for_lstm(dataset,1)
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))

model = deserialize_model("lstm", "beast2")

test_predict = model.predict(test_data)

test_predict = wrapper.scaler.inverse_transform(test_predict)
test_y = wrapper.scaler.inverse_transform([test_y])

A = test_y[0]
F = test_predict[:,0]
evaluate_model(A,F)


size_of_samples = len(F)
aa = [x for x in range(size_of_samples)]



plt.figure(figsize=(20, 6))

plt.plot(df_back.index[0:size_of_samples],A
        [-size_of_samples:], marker='.', label="Naměřená", color='purple', linewidth=2)
plt.plot(df_back.index[0:size_of_samples],
        F[-size_of_samples:], '-', label="Predikce", color='red', linewidth=2)
sns.despine(top=True)
plt.subplots_adjust(left=0.2)
plt.ylabel('kW', size=14)
plt.xlabel('Krok', size=14)
plt.legend(fontsize=16)
plt.show()
