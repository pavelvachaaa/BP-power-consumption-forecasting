import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,\
    mean_squared_error
import time

Y_VALUE_NAME="energy(kWh/hh)"
WEATHER_DEFAULT_COLUMNS = ["temperature","windBearing","dewPoint","windSpeed","pressure","visibility","humidity","time"]

def serialize_model(model: any, type: str ="lstm", code_name="") -> None:
    """
    Funkce serializuje jakýkoliv model využívány v BP
    """
    if code_name == "":
        code_name = int(time.time())
        
    if type=="lstm":
        pass
    elif type=="xgboost":
        filename = f'./out/models/xgboost_model_{code_name}.pkl'
    else:
        pass
    
    pickle.dump(model, open(filename, 'wb'))


def deserialize_model(type: str = "lstm", code_name=""):
    """
    Funkce deserializuje jakýkoliv model využívány v BP
    """  
    if type=="lstm":
        pass
    elif type=="xgboost":
        filename = f'./out/models/xgboost_model_{code_name}.pkl'
    else:
        pass

    return pickle.load(open(filename, "rb"))

def parse_iris_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funkce si vezme dataframe z IRIS
    a přeparsuje ho do použitelného formátu
    """
    df= df.copy()
    df["DateTime"] = df["wstime"]*1000-4070908800000
    df["DateTime"] = (pd.to_datetime( df["DateTime"], unit='ms'))

    # :) 
    df = df.dropna(subset=['value']) 
    df = df[df["value"]!="Null"]
    df = df[df["value"]!=""]

    df = df.set_index('DateTime')

    df[Y_VALUE_NAME] = pd.to_numeric(df["value"])

    df = df.drop(["value"],axis=1)
    df = df.drop(["wstime"],axis=1)

    return df


def load_london_dataset_household(file_path: str, household_id: str, weather_file_path="", weather_columns: list[str] = []) -> pd.DataFrame:
    """
    Funkce dle názvu datové sady načte soubor a vrátí pandas dataframe
    """
    df = pd.read_csv(file_path)
    df = df[df["LCLid"]==household_id]

    df['DateTime'] = pd.to_datetime(df['tstp'])
    df = df[df[Y_VALUE_NAME]!="Null"]
    df[Y_VALUE_NAME] = pd.to_numeric(df[Y_VALUE_NAME])

    df.drop(['tstp'],axis=1,inplace=True)
    df.drop(['LCLid'],axis=1,inplace=True)

    df.set_index('DateTime', inplace=True)
    df.reset_index('DateTime', inplace=True)

    if weather_file_path != "":
        df_weather = pd.read_csv(weather_file_path)
        if len(weather_columns) > 0:
            if ("precipType" in weather_columns):
                label_encoder = LabelEncoder()
                df_weather['precipType'] = label_encoder.fit_transform(df_weather['precipType'])
            df_weather = df_weather[weather_columns]
        else:
            df_weather = df_weather[WEATHER_DEFAULT_COLUMNS]

        df_weather["time"] = pd.to_datetime(df_weather["time"])
        df_weather = df_weather.set_index("time")
        df_weather = df_weather.resample('30T').interpolate().ffill()
        df_weather.reset_index(inplace=True)

        df = pd.merge(df, df_weather, left_on = "DateTime", right_on = "time", how ='left')
        df.set_index('DateTime', inplace=True)

    df = extract_time_features(df)
    return df

def load_iris_dataset(file_path: str, extract_time=True) -> pd.DataFrame:
    """
    Funkce dle názvu datové sady načte soubor a vrátí pandas dataframe
    """
    df = pd.read_json(file_path)
    df = parse_iris_data_frame(df)

    if extract_time:
        df = extract_time_features(df)

    return df


def get_time_features_name() -> list[str]:
    """
    Vrací seznam názvů sloupců, které se extrahují pomocí extract_time_features
    """
    return ["hour", "minute", "dayofweek", "quarter","is_weekend", "month", "year", "dayofyear"]

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funkce extrahuje časové parametry a vloží je separatně do datového rámce
    """
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

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funkce vyvtoří zpožděné proměnné pro XGBoost model
    """
    df = df.copy()

    target_map = df[Y_VALUE_NAME].to_dict()
    # df['energyMean1y'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['energyMean1d'] = (df.index - pd.Timedelta('1 days')).map(target_map)
    df['energyMean7d'] = (df.index - pd.Timedelta('7 days')).map(target_map)
    df['energyMean12h'] = (df.index - pd.Timedelta('12 hours')).map(target_map)
    df['energyMean24h'] = (df.index - pd.Timedelta('24 hours')).map(target_map)
    df['energyMean6h'] = (df.index - pd.Timedelta('6 hours')).map(target_map)
    df['energyMean1h'] = (df.index - pd.Timedelta('1 hours')).map(target_map)
    df["energyMax6h"] = df[Y_VALUE_NAME].rolling(window = 6).max()

    return df

def evaluate_model(A: list[float], F: list[float]) -> None:
    """
    Výpis statistik
    A = actual
    F = forecast
    """
    print('MAE: {:.5f}'.format(mean_absolute_error(A, F)))
    print('MAPE: {:.5f}%'.format(mean_absolute_percentage_error(A, F)*100))
    print('MSE: {:.5f}'.format(mean_squared_error(A, F)))
    print('RMSE: {:.5f}'.format(np.sqrt(mean_squared_error(A, F))))
