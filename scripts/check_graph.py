import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_parquet("data/processed/features.parquet")
df = df.sort_values("ts").reset_index(drop=True)

cut = df["ts"].iloc[int(len(df) * 0.7)]
train = df[df["ts"] < cut]
test = df[df["ts"] >= cut].copy()

K = 0.3                                   # trust decay constant
t_max = train["m_prior_txns"].max()       # normalisers from TRAIN only
u_max = train["m_prior_users"].max()

test["N"] = np.where(test["um_prior"] == 0, 1.0, np.exp(-K * test["um_prior"]))
test["F"] = (1 - test["m_prior_txns"] / t_max).clip(0, 1)
test["C"] = (1 - test["m_prior_users"] / u_max).clip(0, 1)
test["S_graph"] = (test["N"] + test["F"] + test["C"]) / 3

y = test["is_fraud"]
for col in ["N", "F", "C", "S_graph"]:
    print(f"{col}: ROC-AUC {roc_auc_score(y, test[col]):.4f} | "
          f"PR-AUC {average_precision_score(y, test[col]):.4f}")

print("\nShare of test rows that are first-time card-merchant pairs:",
      round((test["um_prior"] == 0).mean(), 4))
print("Mean F, C (test):", round(test["F"].mean(), 4), round(test["C"].mean(), 4))