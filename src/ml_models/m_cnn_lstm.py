import pandas as pd
from core import *


if __name__ == "__main__":
    df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv", "MAC004431", "./data/weather_hourly_darksky.csv", [*WEATHER_DEFAULT_COLUMNS, "precipType"])
    # df: pd.DataFrame = load_london_dataset_household("./data/halfhourly_dataset/halfhourly_dataset/block_0.csv", "MAC004431")
    # df: pd.DataFrame = load_iris_dataset("./data/albistech_dataset/db3.json", "MAC004431")
    
    print(df.head())