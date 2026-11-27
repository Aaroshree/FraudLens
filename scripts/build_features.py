import numpy as np
import pandas as pd
from pathlib import Path

RAW = "data/raw/fraudTrain.csv"
OUT = Path("data/processed/features.parquet")
WINDOWS = {"10m": 600, "1h": 3600, "24h": 86400}  # seconds

df = pd.read_csv(RAW)
df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values("ts", kind="stable").reset_index(drop=True)
df["t"] = df["ts"].astype("int64") // 10**9  # seconds

# ---------- Behavior features ----------
df["hour"] = df["ts"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["log_amt"] = np.log1p(df["amt"])

# ---------- Context proxy ----------
df["context"] = np.where(df["category"].str.endswith("_net"), "QR", "POS")

# ---------- Per-card features (use only PAST rows) ----------
g = df.groupby("cc_num")
df["card_txn_idx"] = g.cumcount()              # 0 for a card's first txn
df["is_cold_start"] = df["card_txn_idx"] < 3
df["gap_sec"] = g["t"].diff()                  # NaN for first txn


def prior_counts(times: pd.Series, window: int) -> np.ndarray:
    """Number of earlier txns of this card within `window` seconds."""
    t = times.values
    start = np.searchsorted(t, t - window, side="left")
    return np.arange(len(t)) - start


for name, secs in WINDOWS.items():
    df[f"vel_{name}"] = g["t"].transform(lambda s, w=secs: prior_counts(s, w))

# ---------- Graph-inspired features (leak-free running counts) ----------
# Raw prior counts only. N, F, C are computed later in the agent,
# using normalisers (t_max, u_max) taken from the TRAIN split.
df["um_prior"] = df.groupby(["cc_num", "merchant"]).cumcount()  # c in N = exp(-k*c)
df["m_prior_txns"] = df.groupby("merchant").cumcount()          # t_m
first_pair = df["um_prior"] == 0                                 # first time this card meets this merchant
df["m_prior_users"] = (
    first_pair.groupby(df["merchant"]).cumsum() - first_pair.astype(int)
)  # u_m: unique cards seen BEFORE this row

# ---------- Save ----------
cols = [
    "ts", "cc_num", "merchant", "category", "context", "is_fraud",
    "amt", "log_amt", "hour", "hour_sin", "hour_cos", "gap_sec",
    "vel_10m", "vel_1h", "vel_24h",
    "card_txn_idx", "is_cold_start",
    "um_prior", "m_prior_txns", "m_prior_users",
]
OUT.parent.mkdir(parents=True, exist_ok=True)
df[cols].to_parquet(OUT, index=False)

print("Saved", OUT, df[cols].shape)
print(df[cols].head())
print("\nContext split:\n", df["context"].value_counts())
print("\nVelocity 99th percentile (legit only):")
print(df.loc[df.is_fraud == 0, ["vel_10m", "vel_1h", "vel_24h"]].quantile(0.99))
print("\nCold-start rows:", int(df["is_cold_start"].sum()))