import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')
from sklearn.preprocessing import LabelEncoder


df_energy = pd.read_csv("./data/daily_dataset.csv")
df_weather = pd.read_csv("./data/weather_daily_darksky.csv")
weather_columns_of_interest = ["temperatureMax","windBearing","dewPoint","cloudCover","windSpeed","pressure","visibility","humidity", "precipType_encoded"
]


df_energy['day'] = pd.to_datetime(df_energy['day'])
df_weather['time'] = pd.to_datetime(df_weather['time'])

label_encoder = LabelEncoder()
df_weather['precipType_encoded'] = label_encoder.fit_transform(df_weather['precipType'])

# Odstranění outlierů ve spotřebě 
df_energy_filtered = df_energy[df_energy['energy_sum'] >= 8.5]
df_energy_filtered.reset_index(drop=True, inplace=True)

# Chceme průměry, abychom data vyčistili zjemnili trend
df_energy_mean = df_energy_filtered.groupby('day')['energy_sum'].mean().reset_index()
df_weather_mean = df_weather.groupby('time')[weather_columns_of_interest].mean().reset_index()

df_combined = pd.merge(df_energy_mean, df_weather_mean, left_on='day', right_on='time', how='inner')

slope, intercept, r_value, p_value, std_err = linregress(df_combined['temperatureMax'], df_combined['energy_sum'])
line = slope * df_combined['temperatureMax'] + intercept

plt.figure(figsize=(12, 6))

sns.scatterplot(x='temperatureMax', y='energy_sum', data=df_combined)

plt.plot(df_combined['temperatureMax'], line, color='red', label='Trendline')

plt.xlabel('Průměrná denní teplota (°C)')
plt.ylabel('Průměrná denní spotřeba [kW/h]')

plt.savefig(f'./out/pv_eda_02_weather_over_consumption.eps', format='eps', bbox_inches='tight', transparent=True)
plt.show()
plt.close()

print(df_combined.head())

fig,ax = plt.subplots(figsize = (9,9))
sns.heatmap(df_combined[["energy_sum",*weather_columns_of_interest]].corr(),ax =ax, annot=True)
plt.savefig(f'./out/pv_eda_02_heat_weather.eps', format='eps', bbox_inches='tight', transparent=True)
plt.show()
plt.close()