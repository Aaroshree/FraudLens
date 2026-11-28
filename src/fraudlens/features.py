from pathlib import Path

import numpy as np
import pandas as pd

COLS = [
    "ts", "cc_num", "merchant", "category", "context", "is_fraud",
    "amt", "log_amt", "hour", "hour_sin", "hour_cos", "gap_sec",
    "vel_10m", "vel_1h", "vel_24h",
    "card_txn_idx", "is_cold_start",
    "um_prior", "m_prior_txns", "m_prior_users",
]


def prior_counts(times, window):
    """Number of earlier txns of this card within `window` seconds."""
    t = times.values
    start = np.searchsorted(t, t - window, side="left")
    return np.arange(len(t)) - start


def build_features(cfg, raw_csv, save_to=None):
    f = cfg["features"]
    df = pd.read_csv(raw_csv)
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["t"] = (df["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

    df["hour"] = df["ts"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["log_amt"] = np.log1p(df["amt"])

    suffix = cfg["context_rule"]["qr_category_suffix"]
    df["context"] = np.where(df["category"].str.endswith(suffix), "QR", "POS")

    g = df.groupby("cc_num")
    df["card_txn_idx"] = g.cumcount()
    df["is_cold_start"] = df["card_txn_idx"] < f["cold_start_txns"]
    df["gap_sec"] = g["t"].diff()
    for name, secs in f["windows_sec"].items():
        df[name] = g["t"].transform(lambda s, w=secs: prior_counts(s, w))

    df["um_prior"] = df.groupby(["cc_num", "merchant"]).cumcount()
    df["m_prior_txns"] = df.groupby("merchant").cumcount()
    first_pair = df["um_prior"] == 0
    df["m_prior_users"] = (
        first_pair.groupby(df["merchant"]).cumsum() - first_pair.astype(int)
    )

    out = df[COLS]
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(save_to, index=False)
    return out