import pandas as pd
from core import *
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import dates as mdates
import locale
import datetime
from statsmodels.tsa.stattools import kpss, adfuller
locale.setlocale(locale.LC_ALL, 'cs_CZ')
sns.set_context("paper", font_scale=1.5)
sns.set_style('white')
import os
is_albistech = False


def plot_acf_pacf(data):
    adf_stat, adf_p_value, _, _, _,_,= adfuller(data['energy(kWh/hh)'])
    data_diff = data.diff().dropna() # 1. diference
    adf_stat_after, adf_p_value_after, _, _, _,_,= adfuller(data_diff['energy(kWh/hh)'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    fmt_month = mdates.MonthLocator()
    fmt_year = mdates.YearLocator()
    ax1.xaxis.set_minor_locator(fmt_month)
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(fmt_year)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.tick_params(labelsize=13, which='both')
    sec_xaxis = ax1.secondary_xaxis(-0.2)
    sec_xaxis.xaxis.set_major_locator(fmt_year)
    sec_xaxis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    sec_xaxis.spines['bottom'].set_visible(False)
    sec_xaxis.tick_params(length=0, labelsize=13)

    ax1.plot(data_diff.index, data_diff[Y_VALUE_NAME])
    ax1.set_title(f'Spotřeba energie po první diferenci')
    ax1.set_xlabel('Datum')
    ax1.set_ylabel('Spotřeba [kWh]')

    plot_acf(data_diff[Y_VALUE_NAME], ax=ax2, lags=100, auto_ylims=True)
    ax2.set_title('Autokorelační funkce')
    ax2.set_xlabel('Zpoždění')
    ax2.set_ylabel('Autokorelační koeficient')

    plot_pacf(data_diff[Y_VALUE_NAME], ax=ax3, lags=100,  auto_ylims=True)
    ax3.set_title('Parcialní autokorelační funkce')
    ax3.set_xlabel('Zpoždění')
    ax3.set_ylabel('Parcialní korelační koeficient')

    plt.tight_layout()
    plt.savefig('./out/arima_pacf_acf.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == "__main__":
    df: pd.DataFrame = load_agg_dataseet("./data/agg_halfhourly.csv")
    df_filtered = df.loc['2013-01':'2014-01', :]

    adf_stat, adf_p_value, _, _, _,_,= adfuller(df_filtered['energy(kWh/hh)'])
 
    print(adf_stat)
    print(adf_p_value)
    plot_acf_pacf(df_filtered)


