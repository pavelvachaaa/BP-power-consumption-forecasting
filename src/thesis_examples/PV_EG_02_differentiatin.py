import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

def do_diff(file_name: str) -> None:
    """
    Funkce slouží jako materiál pro kapitolu - stacionární řady
    """
    df = pd.read_csv(file_name, delimiter=";")
    df["Value"] = pd.to_numeric(df["Value"])

    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))

    df.set_index('Date', inplace=True)
  
    # 1. diference
    df['First_Diff'] = df['Value'].diff()

    # Plot the original data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Value'], label='Míra nezaměstnanosti', linewidth=2)
    # plt.title('Registrovaná míra nezaměstnanosti')
    plt.xlabel('Čas')
    plt.ylabel('Míra nezaměstnanosti [%]')
    plt.legend()
    plt.savefig('pv_eg_02_mira_nezam.eps', format='eps', bbox_inches='tight', transparent=True)

    plt.show()


 

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['First_Diff'], label='Míra nezaměstnanosti', color='orange', linewidth=2)
    # plt.title('1. Diference')
    plt.xlabel('Čas')
    plt.ylabel('Míra nezaměstnanosti')
    plt.legend()
    plt.savefig('pv_eg_02_mira_nezam_diff.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()


   # Undo the first difference
    df['Undone_First_Diff'] = df['First_Diff'].cumsum()

    # Plot the data with first difference undone
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Undone_First_Diff'], label='Undone First Difference', color='orange', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.savefig('undone_first_diff_plot.eps', format='eps', bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == "__main__":
    do_diff("./data/nezamestnanost_cr_stat_urad.csv")