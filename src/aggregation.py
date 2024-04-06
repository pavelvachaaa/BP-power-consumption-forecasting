import pandas as pd
import os
is_albistech = False

if __name__ == "__main__":
    
    directory_path = "./data/halfhourly_dataset/halfhourly_dataset/"
    all_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]
    all_data = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(directory_path, file)
        data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, data])

    all_data['tstp'] = pd.to_datetime(all_data['tstp'])
    all_data= all_data[all_data["energy(kWh/hh)"]!="Null"]
    all_data["energy(kWh/hh)"] = pd.to_numeric(all_data["energy(kWh/hh)"])


    new_df = all_data.groupby("tstp")["energy(kWh/hh)"].mean()
    new_df.to_csv("./data/agg_halfhourly.csv")    
    
    print(new_df)  