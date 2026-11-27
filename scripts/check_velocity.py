import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("data/processed/features.parquet")

TH = {"s": 1, "m": 2, "l": 14}          # flag if count > threshold
V = {"s": 0.40, "m": 0.40, "l": 0.0}   # severity weights
LAM = 0.30                              # amplification when all 3 fire

fs = (df["vel_10m"] > TH["s"]).astype(int)
fm = (df["vel_1h"] > TH["m"]).astype(int)
fl = (df["vel_24h"] > TH["l"]).astype(int)

df["S_velocity"] = np.minimum(
    1, V["s"] * fs + V["m"] * fm + V["l"] * fl + LAM * fs * fm * fl
)

print("Flag rate by class:")
print(pd.DataFrame({"f_short": fs, "f_med": fm, "f_long": fl,
                    "is_fraud": df["is_fraud"]}).groupby("is_fraud").mean().round(4))
print("\nMean S_velocity by class:")
print(df.groupby("is_fraud")["S_velocity"].mean().round(4))
print("\nVelocity-only AUC:", round(roc_auc_score(df["is_fraud"], df["S_velocity"]), 4))
print("\nAUC by context:")
for c, d in df.groupby("context"):
    print(c, round(roc_auc_score(d["is_fraud"], d["S_velocity"]), 4))