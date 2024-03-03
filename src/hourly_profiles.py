import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')
import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')


directory_path = "./data/halfhourly_dataset/halfhourly_dataset/"
all_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

all_data = pd.DataFrame()

for file in all_files:
    file_path = os.path.join(directory_path, file)
    data = pd.read_csv(file_path)
    all_data = pd.concat([all_data, data])

all_data['tstp'] = pd.to_datetime(all_data['tstp'])
all_data= all_data[all_data["energy(kWh/hh)"]!="Null"]
all_data["energy(kWh/hh)"] = pd.to_numeric(all_data["energy(kWh/hh)"])

hourly_average = all_data.groupby(all_data['tstp'].dt.hour)['energy(kWh/hh)'].mean().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(hourly_average['tstp'], hourly_average['energy(kWh/hh)'], marker='o', linestyle='-', color='r')
plt.title('Průměrná hodinová spotřeba energie')
plt.xlabel('Hodina')
plt.ylabel('Spotřeba [kW/h]')
plt.grid(True)


plt.savefig(f'./out/pv_eda_01_week_avg_hourly_consumption_all.eps', format='eps', bbox_inches='tight', transparent=True)
plt.show()
plt.close()
