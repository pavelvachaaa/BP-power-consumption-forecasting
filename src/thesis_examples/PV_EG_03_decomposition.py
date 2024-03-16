import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")


def do_moving_average_decomposition(file_name: str) -> None:
    """
    Funkce slouží jako materiál pro kapitolu - klasická dekompozice
    Řada vypadá aditivně, takže zvolíme aditivní model.
    """
    df = pd.read_csv(file_name, delimiter=";")
    df["Value"] = pd.to_numeric(df["Value"])

    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

    df.set_index('Date', inplace=True)

    result = seasonal_decompose(df['Value'], model='additive', period=12)  
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    
    result.observed.plot(ax=ax1, linewidth=2)
    ax1.set_ylabel('Původní')
    ax1.set_xlabel('')
    
    result.trend.plot(ax=ax2,linewidth=2)
    ax2.set_ylabel('Trendová')
    ax2.set_xlabel('')
    
    result.seasonal.plot(ax=ax3,linewidth=2)
    ax3.set_ylabel('Sezónní')
    ax3.set_xlabel('')
    
    result.resid.plot(ax=ax4, linewidth=2)
    ax4.set_ylabel('Náhodná')
    ax4.set_xlabel('Rok')


    plt.savefig('pv_eg_03_classical_decomposition.eps', format='eps', bbox_inches='tight', transparent=True)

    plt.show()


    plt.show()
if __name__ == "__main__":
    do_moving_average_decomposition("../data/nezamestnanost_cr_stat_urad.csv")



