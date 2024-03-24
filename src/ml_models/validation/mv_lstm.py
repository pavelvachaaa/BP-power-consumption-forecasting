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
DAYS_OF_PREDICTION =  ONE_DAY*5

# Extrahujeme časové features
df = wrapper.add_lags(df, Y_VALUE_NAME)

df= df.iloc[-DAYS_OF_PREDICTION:].copy()
df_back = df

# Ujistéme se, že naše Y_VAL je float64
df = df[Y_VALUE_NAME].values.astype('float64')

# Provedeme transformaci hodnot dataframu
dataset = wrapper.transform_for_lstm(df)

test_data, test_y = wrapper.to_sequence_for_lstm(dataset,24)
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))

model = deserialize_model("lstm", "beast")

test_predict = model.predict(test_data)

test_predict = wrapper.scaler.inverse_transform(test_predict)
test_y = wrapper.scaler.inverse_transform([test_y])

A = test_y[0]
F = test_predict[:,0]
evaluate_model(A,F)


size_of_samples = len(F)
aa = [x for x in range(size_of_samples)]



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
