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
    energy_data = pd.read_csv(file_name)

    # Výběr domácnosti
    df = energy_data[energy_data["LCLid"] == "MAC000246"]

    # Převod energie na číslo
    df = df[df["energy(kWh/hh)"] != "Null"]
    df["energy(kWh/hh)"] = pd.to_numeric(df["energy(kWh/hh)"])

    # Vytvoření indexu na čas
    df["DateTime"] = pd.to_datetime(df["tstp"], infer_datetime_format=True)
    df = df[df["DateTime"].dt.year == 2013]
    df = df[ df["DateTime"].dt.month == 6 ]
    df = df[ df["DateTime"].dt.day <= 21 ]


    df.drop(['tstp'], axis=1, inplace=True)
    df = df.set_index('DateTime')

    df['MovingAverage'] = df["energy(kWh/hh)"].rolling(30).mean()

    fig, ax = plt.subplots(figsize=(20,10))

    df["energy(kWh/hh)"].plot(ax=ax, linewidth=1, label="Spotřeba energie")
    df['MovingAverage'].plot(ax=ax, label=f'Klouzavý průmer řádu m=30)', color='red', linewidth=2)

    plt.ylabel('Spotřeba [kWh/hh]', size=14)
    plt.xlabel('Den v měsíci', size=14)

    # Adding a legend with font size of 15
    plt.legend(fontsize=16)

    plt.savefig('pv_eg_01_moving_average.eps', format='eps', bbox_inches='tight', transparent=True)




if __name__ == "__main__":
    do_moving_average_decomposition("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv")


# Selecting three houses having maximum number of data points
# df = energy_data[energy_data["LCLid"] == "MAC000246"]
