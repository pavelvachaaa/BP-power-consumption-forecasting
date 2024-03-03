import pandas as pd

def parse_iris_data_frame(df: pd.DataFrame):
    """
    Funkce si vezme dataframe z IRIS
    a přeparsuje ho do použitelného formátu
    """
    df= df.copy()
    df["DateTime"] = df["wstime"]*1000-4070908800000
    df["DateTime"] = (pd.to_datetime( df["DateTime"], unit='ms'))
    df = df.dropna(subset=['value'])
    df = df[df["value"]!="Null"]
    df = df[df["value"]!=""]

    df = df.set_index('DateTime')

    df["energy(kWh/hh)"] = pd.to_numeric(df["value"])

    df = df.drop(["value"],axis=1)
    df = df.drop(["wstime"],axis=1)

    return df

def load_iris_dataset(file_path: str) -> pd.DataFrame :
    """
    Funkce dle názvu datové sady načte soubor a vrátí pandas dataframe
    """
    df = pd.read_json(file_path)
    return parse_iris_data_frame(df)
