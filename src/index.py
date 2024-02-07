import pandas as pd
import numpy as np

energy_data = pd.read_csv("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv")
print(energy_data.tail(100))


# # Weather Data
# weather_data = pd.read_csv("./data_new/weather_hourly_darksky.csv")
# energy_data[energy_data["energy(kWh/hh)"] == "Null"]
# energy_data = energy_data[energy_data["energy(kWh/hh)"] != "Null"]

# energy_data.groupby("LCLid").count().sort_values(by=['energy(kWh/hh)'], ascending=False)
# Selecting three houses having maximum number of data points
# df = energy_data[energy_data["LCLid"] == "MAC000246"]
