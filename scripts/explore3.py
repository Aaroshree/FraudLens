import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/fraudTrain.csv", parse_dates=["trans_date_trans_time"])
df["hour"] = df["trans_date_trans_time"].dt.hour


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


# 1. Impossible travel between consecutive merchants
df = df.sort_values(["cc_num", "trans_date_trans_time"])
g = df.groupby("cc_num")
prev_lat = g["merch_lat"].shift()
prev_lon = g["merch_long"].shift()
km = hav(prev_lat, prev_lon, df["merch_lat"], df["merch_long"])
hrs = (g["trans_date_trans_time"].diff().dt.total_seconds() / 3600).clip(lower=1/60)
df["speed_kmh"] = km / hrs
print(df.groupby("is_fraud")["speed_kmh"].describe())

# 2. Night share of legit vs fraud
night = df["hour"].isin([22, 23, 0, 1, 2, 3])
print("Share of legit txns at night:", round(night[df.is_fraud == 0].mean(), 4))
print("Share of fraud txns at night:", round(night[df.is_fraud == 1].mean(), 4))