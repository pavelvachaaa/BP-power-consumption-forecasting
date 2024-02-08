import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")


def do_moving_average_decomposition(file_name: str) -> None:
    """
    Funkce slouží jako materiál pro kapitolu - dekompozice pomocí klouzavého průměru
    """
    df = pd.read_csv(file_name, delimiter=";")
    df["Value"] = pd.to_numeric(df["Value"])

    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

    df.set_index('Date', inplace=True)

    df['MovingAverage'] = df["Value"].rolling(30).mean()

    fig, ax = plt.subplots(figsize=(12,6))

    df["Value"].plot(ax=ax, linewidth=1, label="Míra nezaměstnanosti")
    df['MovingAverage'].plot(ax=ax, label=f'Klouzavý průmer řádu m=30)', color='red', linewidth=2)

    plt.ylabel('Míra nezaměstnanosti [%]', size=14)
    plt.xlabel('Den v měsíci', size=14)
    plt.legend(fontsize=16)

    plt.savefig('pv_eg_01_moving_average.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == "__main__":
    do_moving_average_decomposition("./data/nezamestnanost_cr_stat_urad.csv")



