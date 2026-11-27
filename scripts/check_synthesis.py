import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
df = df.sort_values("ts").reset_index(drop=True)
df["log_gap"] = np.log1p(df["gap_sec"].fillna(df["gap_sec"].median()))
cut = df["ts"].iloc[int(len(df) * 0.7)]
test = df[df["ts"] >= cut].copy()

# Behavior scores saved by check_behavior.py (same split, same order)
beh = pd.read_parquet("data/processed/behavior_scores_test.parquet")
assert len(beh) == len(test)
assert (beh["cc_num"].values == test["cc_num"].values).all()
assert (beh["is_fraud"].values == test["is_fraud"].values).all()
test["S_if"] = beh["S_if"].values
test["S_gb"] = beh["S_gb"].values

# Agents
fs = (test["vel_10m"] > 1).astype(int)
fm = (test["vel_1h"] > 2).astype(int)
test["S_vel"] = np.minimum(1, 0.40 * fs + 0.40 * fm)
test["S_gra"] = np.where(test["um_prior"] == 0, 1.0, np.exp(-0.3 * test["um_prior"]))

# Paper weights with Geo removed, renormalised so each context sums to 1
PAPER = {"POS": {"vel": 0.40, "beh": 0.20, "gra": 0.15},
         "QR": {"vel": 0.30, "beh": 0.25, "gra": 0.20}}
W = {c: {k: v / sum(w.values()) for k, v in w.items()} for c, w in PAPER.items()}
print("Weights used:", {c: {k: round(v, 3) for k, v in w.items()} for c, w in W.items()})


def composite(d, beh_col):
    wv = d["context"].map(lambda c: W[c]["vel"])
    wb = d["context"].map(lambda c: W[c]["beh"])
    wg = d["context"].map(lambda c: W[c]["gra"])
    return wv * d["S_vel"] + wb * d[beh_col] + wg * d["S_gra"]


# Time split inside the test period: calibrate thresholds on first half, evaluate on second
mid = test["ts"].iloc[len(test) // 2]
val, ev = test[test["ts"] < mid].copy(), test[test["ts"] >= mid].copy()
PAPER_TH = {"POS": [0.50, 0.70, 0.90], "QR": [0.55, 0.75, 0.90]}
NAMES = ["Approve", "Single OTP", "Dual OTP", "Block"]


def decide(d, R, th):
    out = np.zeros(len(d), dtype=int)
    for c in ["POS", "QR"]:
        m = (d["context"] == c).values
        out[m] = np.digitize(R[m], th[c])
    return out


def metrics(d, dec, label):
    y = d["is_fraud"].values
    esc = dec > 0
    tp = (esc & (y == 1)).sum()
    prec = tp / max(esc.sum(), 1)
    rec = tp / max((y == 1).sum(), 1)
    fpr = (esc & (y == 0)).sum() / max((y == 0).sum(), 1)
    print(f"  {label}: precision {prec:.3f} | recall {rec:.3f} | FPR {fpr:.3%}")


for beh_col, name in [("S_if", "Paper agents (Isolation Forest)"),
                      ("S_gb", "With supervised Behavior (extension)")]:
    val["R"] = composite(val, beh_col).values
    ev["R"] = composite(ev, beh_col).values

    cal = {c: np.quantile(val.loc[(val["context"] == c) & (val["is_fraud"] == 0), "R"],
                          [0.95, 0.99, 0.999]) for c in ["POS", "QR"]}
    print(f"\n=== {name} ===")
    print("Calibrated thresholds:", {c: np.round(v, 3).tolist() for c, v in cal.items()})

    for tname, th in [("Paper thresholds", PAPER_TH), ("Calibrated thresholds", cal)]:
        dec = decide(ev, ev["R"].values, th)
        print(f"{tname}")
        metrics(ev, dec, "ALL")
        for c in ["POS", "QR"]:
            m = (ev["context"] == c).values
            metrics(ev[m], dec[m], c)
        dist = pd.crosstab(ev["is_fraud"], dec).reindex(columns=range(4), fill_value=0)
        dist.columns = NAMES
        print((dist.div(dist.sum(axis=1), axis=0) * 100).round(1).to_string())