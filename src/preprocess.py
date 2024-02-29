import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import locale
import datetime
locale.setlocale(locale.LC_ALL,'cs_CZ')

import seaborn as sns
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')

df = pd.read_csv("./data/daily_dataset.csv")
days = []

df_daily_selection=df[df["energy_count"]==48][["LCLid","day"]].dropna()
days.append(df_daily_selection)

df_dailyselection=pd.concat(days,axis=0)
df_dailyselection["day"]=pd.to_datetime(df_dailyselection["day"])
df_count=df_dailyselection.groupby(["day"]).count()
print(df_count)
fig,ax=plt.subplots(figsize=(10,6))
df_count.plot(ax=ax)
plt.xlabel("Den")
plt.ylabel("Počet domácností")
fig.tight_layout()

plt.savefig('./out/pv_eda_01_households_over_time.eps', format='eps', bbox_inches='tight', transparent=True)

plt.show()

plt.close()


start_date=datetime.datetime(year=2013,month=1,day=1)
end_date=datetime.datetime(year=2014,month=1,day=1)
df_dailyselection_zoom=df_dailyselection[(df_dailyselection["day"]>=start_date) & (df_dailyselection["day"]<end_date)]
df_countperid=df_dailyselection_zoom.groupby(["LCLid"]).count()


fig,ax=plt.subplots(figsize=(10,6))
sns.boxplot(x=df_countperid["day"], ax=ax)
plt.xlim(355,370)
plt.xlabel("Počet zaznamenaných dní v roce 2013")
fig.tight_layout()
 
plt.savefig("./out/number_of_records_distribution.eps", format='eps', bbox_inches='tight', transparent=True)

plt.show()

list_devices=list(df_countperid[df_countperid>357].index)
print("Households for the rest of the study: {} households".format(len(list_devices)))