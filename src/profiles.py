import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

def extract_features(df: pd.DataFrame):
    df = df.copy()
    df["hour"]= df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"]=df.index.dayofweek # sunday = 6
    df["quarter"] = df.index.quarter
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df["week_day"] = df.index.day_name()

    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    
    return df

df = pd.read_csv("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv")

weather_data = pd.read_csv("./data/weather_hourly_darksky.csv")
weather_data["time"] = pd.to_datetime(weather_data["time"])
weather_data = weather_data.set_index("time")
weather_data = weather_data.resample('30T').interpolate().ffill()
weather_data.drop(['icon','summary'],axis=1,inplace=True)

df= df[df["energy(kWh/hh)"]!="Null"]

df["energy(kWh/hh)"] = pd.to_numeric(df["energy(kWh/hh)"])
df["DateTime"] = pd.to_datetime(df["tstp"])
df.drop(['tstp'],axis=1,inplace=True)

# Výběr tří domácností na základě toho, jestli jsou všechny měření ze dne dostupná.
df.groupby("LCLid").count().sort_values(by=['energy(kWh/hh)'], ascending=False)

house_a = df[df["LCLid"]=="MAC000246"]
house_b = df[df["LCLid"]=="MAC004387"]
house_c = df[df["LCLid"]=="MAC004431"]
house_d = load_iris_dataset("./data/albistech_dataset/db_2548_1683974964.2201283.json")

house_a = house_a.drop(["LCLid"],axis=1)
house_b = house_b.drop(["LCLid"],axis=1)
house_c = house_c.drop(["LCLid"],axis=1)

house_a = house_a.set_index('DateTime')
house_b = house_b.set_index('DateTime')
house_c = house_c.set_index('DateTime')

house_a = house_a.reset_index()
house_b = house_b.reset_index()
house_c = house_c.reset_index()

weather_data = weather_data.reset_index()

house_a = pd.merge(house_a , weather_data, left_on = "DateTime", right_on = "time", how ='left')
house_b = pd.merge(house_b , weather_data, left_on = "DateTime", right_on = "time", how ='left')
house_c = pd.merge(house_c , weather_data, left_on = "DateTime", right_on = "time", how ='left')

house_a = house_a.set_index('DateTime')
house_b = house_b.set_index('DateTime')
house_c = house_c.set_index('DateTime')

house_a = extract_features(house_a)
house_b = extract_features(house_b)
house_c = extract_features(house_c)
house_d= extract_features(house_d)

print(house_a.head())
print(house_d.head())

is_weekend = 0
a_group = house_a[['hour',"energy(kWh/hh)", "is_weekend","quarter"]].loc[lambda x: x.is_weekend == is_weekend].groupby('hour').mean().reset_index()
b_group = house_b[['hour',"energy(kWh/hh)", "is_weekend","quarter"]].loc[lambda x: x.is_weekend == is_weekend].groupby('hour').mean().reset_index()
c_group = house_c[['hour',"energy(kWh/hh)", "is_weekend","quarter"]].loc[lambda x: x.is_weekend == is_weekend].groupby('hour').mean().reset_index()
d_group = house_d[['hour',"energy(kWh/hh)", "is_weekend", "dayofweek","quarter"]].loc[lambda x: x.is_weekend == is_weekend].groupby('hour').mean().reset_index()
    
# create data frames
ha= pd.DataFrame({"energy(kWh/hh)":a_group["energy(kWh/hh)"].values,"house":"Rezidence - MAC000246","hour":a_group["hour"]})
hb= pd.DataFrame({"energy(kWh/hh)":b_group["energy(kWh/hh)"].values,"house":"Rezidence - MAC004387","hour":b_group["hour"]})
hc= pd.DataFrame({"energy(kWh/hh)":c_group["energy(kWh/hh)"].values,"house":"Rezidence - MAC004431","hour":c_group["hour"]})
hd= pd.DataFrame({"energy(kWh/hh)":d_group["energy(kWh/hh)"].values,"house":"Poštovská 3 - byt 5.02","hour":d_group["hour"]})
all=pd.concat([ha,hb,hc,hd])

fig, ax = plt.subplots(figsize=(13,5))

ax.plot(ha["hour"], ha["energy(kWh/hh)"], label=ha["house"][0])
ax.plot(hb["hour"], hb["energy(kWh/hh)"], label=hb["house"][0])
ax.plot(hc["hour"], hc["energy(kWh/hh)"], label=hc["house"][0])
ax.plot(hd["hour"], hd["energy(kWh/hh)"], label=hd["house"][0])

# Calculate standard deviation
std_dev = all.groupby('hour')['energy(kWh/hh)'].std()

# Plot error bars representing +/- 3 std
plt.errorbar(std_dev.index, all.groupby('hour')['energy(kWh/hh)'].mean(), yerr=3*std_dev, fmt='o', alpha=0.5)


plt.xlabel('Hodina')
plt.ylabel('Spotřeba (kWh)')
# plt.title(f'Hodinová spotřeba čtyř vybraných budov {"o víkendu" if is_weekend else  "v pracovním týdnu"}')
plt.xticks(range(24))

plt.xlim(left=0,right=23)

plt.legend()

plt.savefig("week_avg_hour.eps", format="eps")

plt.savefig(f'./out/pv_eda_01_week_avg_hour_group_weekend_{is_weekend}.eps', format='eps', bbox_inches='tight', transparent=True)
plt.show()
plt.close()



#########################

a_group = house_a[['week_day',"energy(kWh/hh)", "is_weekend","quarter"]].groupby('week_day').mean().reset_index()
b_group = house_b[['week_day',"energy(kWh/hh)", "is_weekend","quarter"]].groupby('week_day').mean().reset_index()
c_group = house_c[['week_day',"energy(kWh/hh)", "is_weekend","quarter"]].groupby('week_day').mean().reset_index()
d_group = house_d[['week_day',"energy(kWh/hh)","dayofweek","quarter"]].groupby('week_day').mean().reset_index()
    
ha= pd.DataFrame({"energy(kWh/hh)":a_group["energy(kWh/hh)"].values,"house":"Rezidence - MAC000246","week_day":a_group["week_day"]})
hb= pd.DataFrame({"energy(kWh/hh)":b_group["energy(kWh/hh)"].values,"house":"Rezidence - MAC004387","week_day":b_group["week_day"]})
hc= pd.DataFrame({"energy(kWh/hh)":c_group["energy(kWh/hh)"].values,"house":"Rezidence - MAC004431","week_day":c_group["week_day"]})
hd= pd.DataFrame({"energy(kWh/hh)":d_group["energy(kWh/hh)"].values,"house":"Poštovská 3 - byt 5.02","dayofweek":d_group["week_day"]})
all=pd.concat([ha,hb,hc,hd])

fig, ax = plt.subplots(figsize=(13,5))
x = np.array([0,1,2,3,4,5,6])
my_xticks = ha["week_day"].replace("Saturday", "Sobota").replace("Friday", "Pátek").replace("Monday", "Pondělí").replace("Tuesday", "Úterý").replace("Thursday", "Čtvrtek").replace("Sunday", "Neděle").replace("Wednesday", "Středa")

ax.plot(x,ha["energy(kWh/hh)"], label=ha["house"][0])
ax.plot(x, hb["energy(kWh/hh)"], label=hb["house"][0])
ax.plot(x, hc["energy(kWh/hh)"], label=hc["house"][0])
ax.plot(x, hd["energy(kWh/hh)"], label=hd["house"][0])


plt.xticks(x, my_xticks)
plt.xlabel('Den')
plt.ylabel('Spotřeba (kWh)')
# plt.title('Průměrná týdenní spotřeba čtyř vybraných budov')

plt.xticks(range(7))

plt.xlim(left=0,right=6)
plt.legend()


plt.savefig(f'./out/pv_eda_01_week_avg_daily_consumption_group.eps', format='eps', bbox_inches='tight', transparent=True)
plt.show()
plt.close()

#####

ha_quarter = house_a[['hour', 'quarter', 'energy(kWh/hh)']].groupby(['hour', 'quarter']).mean().reset_index()
hb_quarter = house_b[['hour', 'quarter', 'energy(kWh/hh)']].groupby(['hour', 'quarter']).mean().reset_index()
hc_quarter = house_c[['hour', 'quarter', 'energy(kWh/hh)']].groupby(['hour', 'quarter']).mean().reset_index()
hd_quarter = house_d[['hour', 'quarter', 'energy(kWh/hh)']].groupby(['hour', 'quarter']).mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'hspace': 0.4})

axes[0, 0].plot(ha_quarter[ha_quarter['quarter'] == 1]['hour'], ha_quarter[ha_quarter['quarter'] == 1]['energy(kWh/hh)'], label='Jaro')
axes[0, 0].plot(ha_quarter[ha_quarter['quarter'] == 2]['hour'], ha_quarter[ha_quarter['quarter'] == 2]['energy(kWh/hh)'], label='Léto')
axes[0, 0].plot(ha_quarter[ha_quarter['quarter'] == 3]['hour'], ha_quarter[ha_quarter['quarter'] == 3]['energy(kWh/hh)'], label='Podzim')
axes[0, 0].plot(ha_quarter[ha_quarter['quarter'] == 4]['hour'], ha_quarter[ha_quarter['quarter'] == 4]['energy(kWh/hh)'], label='Zima')
axes[0, 0].set_title('Rezidence - MAC000246')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Hodina')
axes[0, 0].set_ylabel('Spotřeba (kWh)')

axes[0, 1].plot(hb_quarter[hb_quarter['quarter'] == 1]['hour'], hb_quarter[hb_quarter['quarter'] == 1]['energy(kWh/hh)'], label='Jaro')
axes[0, 1].plot(hb_quarter[hb_quarter['quarter'] == 2]['hour'], hb_quarter[hb_quarter['quarter'] == 2]['energy(kWh/hh)'], label='Léto')
axes[0, 1].plot(hb_quarter[hb_quarter['quarter'] == 3]['hour'], hb_quarter[hb_quarter['quarter'] == 3]['energy(kWh/hh)'], label='Podzim')
axes[0, 1].plot(hb_quarter[hb_quarter['quarter'] == 4]['hour'], hb_quarter[hb_quarter['quarter'] == 4]['energy(kWh/hh)'], label='Zima')
axes[0, 1].set_title('Rezidence - MAC004387')
axes[0, 1].legend()
axes[0, 1].set_xlabel('Hodina')
axes[0, 1].set_ylabel('Spotřeba (kWh)')

axes[1, 0].plot(hc_quarter[hc_quarter['quarter'] == 1]['hour'], hc_quarter[hc_quarter['quarter'] == 1]['energy(kWh/hh)'], label='Jaro')
axes[1, 0].plot(hc_quarter[hc_quarter['quarter'] == 2]['hour'], hc_quarter[hc_quarter['quarter'] == 2]['energy(kWh/hh)'], label='Léto')
axes[1, 0].plot(hc_quarter[hc_quarter['quarter'] == 3]['hour'], hc_quarter[hc_quarter['quarter'] == 3]['energy(kWh/hh)'], label='Podzim')
axes[1, 0].plot(hc_quarter[hc_quarter['quarter'] == 4]['hour'], hc_quarter[hc_quarter['quarter'] == 4]['energy(kWh/hh)'], label='Zima')
axes[1, 0].set_title('Rezidence - MAC004431')
axes[1, 0].legend()
axes[1, 0].set_xlabel('Hodina')
axes[1, 0].set_ylabel('Spotřeba (kWh)')

axes[1, 1].plot(hd_quarter[hd_quarter['quarter'] == 1]['hour'], hd_quarter[hd_quarter['quarter'] == 1]['energy(kWh/hh)'], label='Jaro')
axes[1, 1].plot(hd_quarter[hd_quarter['quarter'] == 2]['hour'], hd_quarter[hd_quarter['quarter'] == 2]['energy(kWh/hh)'], label='Léto')
axes[1, 1].plot(hd_quarter[hd_quarter['quarter'] == 3]['hour'], hd_quarter[hd_quarter['quarter'] == 3]['energy(kWh/hh)'], label='Podzim')
axes[1, 1].plot(hd_quarter[hd_quarter['quarter'] == 4]['hour'], hd_quarter[hd_quarter['quarter'] == 4]['energy(kWh/hh)'], label='Zima')
axes[1, 1].set_title('Poštovská 3 - byt 5.02')
axes[1, 1].legend()
axes[1, 1].set_xlabel('Hodina')
axes[1, 1].set_ylabel('Spotřeba (kWh)')


# fig.suptitle('Hodinová spotřeba čtyř vybraných budov podle ročního období')

plt.savefig('./out/hourly_energy_consumption_per_quarter.eps', format='eps', bbox_inches='tight', transparent=True)
plt.show()