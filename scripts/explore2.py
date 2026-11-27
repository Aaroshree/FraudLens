import pandas as pd

df = pd.read_csv("data/raw/fraudTrain.csv")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
import numpy as np

df["hour"] = df["trans_date_trans_time"].dt.hour
print(df.groupby("hour")["is_fraud"].mean().round(4))

def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))

df["dist_km"] = hav(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
print(df.groupby("is_fraud")["dist_km"].describe())

df = df.sort_values(["cc_num", "trans_date_trans_time"])
gap = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()
print(gap.groupby(df["is_fraud"]).describe())
print(df.columns.tolist())