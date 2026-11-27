import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_parquet("data/processed/features.parquet")
df = df.sort_values("ts").reset_index(drop=True)
cut = df["ts"].iloc[int(len(df) * 0.7)]
test = df[df["ts"] >= cut].copy()

beh = pd.read_parquet("data/processed/behavior_scores_test.parquet")
test["S_if"] = beh["S_if"].values
test["S_gb"] = beh["S_gb"].values
fs = (test["vel_10m"] > 1).astype(int)
fm = (test["vel_1h"] > 2).astype(int)
test["S_vel"] = np.minimum(1, 0.40 * fs + 0.40 * fm)
test["S_gra"] = np.where(test["um_prior"] == 0, 1.0, np.exp(-0.3 * test["um_prior"]))

mid = test["ts"].iloc[len(test) // 2]
val, ev = test[test["ts"] < mid].copy(), test[test["ts"] >= mid].copy()

FIXED = {"POS": np.array([0.533, 0.267, 0.200]),   # paper weights, Geo removed
         "QR": np.array([0.400, 0.333, 0.267])}     # order: vel, behavior, graph


def learn(d, feats):
    out = {}
    for c in ["POS", "QR"]:
        m = d[d["context"] == c]
        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(m[feats], m["is_fraud"])
        w = np.clip(lr.coef_[0], 0, None)
        out[c] = w / w.sum()
    return out


def score(d, W, feats):
    R = np.zeros(len(d))
    for c in ["POS", "QR"]:
        m = (d["context"] == c).values
        R[m] = d.loc[m, feats].values @ W[c]
    return R


def evaluate(name, Rval, Rev):
    esc = np.zeros(len(ev), dtype=bool)
    for c in ["POS", "QR"]:
        mv = ((val["context"] == c) & (val["is_fraud"] == 0)).values
        me = (ev["context"] == c).values
        th = np.quantile(Rval[mv], 0.95)           # about 5% FPR on legit
        esc[me] = Rev[me] > th
    y = ev["is_fraud"].values
    rec = (esc & (y == 1)).sum() / (y == 1).sum()
    fpr = (esc & (y == 0)).sum() / (y == 0).sum()
    print(f"{name}: ROC-AUC {roc_auc_score(y, Rev):.4f} | "
          f"PR-AUC {average_precision_score(y, Rev):.4f} | "
          f"recall {rec:.3f} @ FPR {fpr:.2%}")


for label, feats in [("Paper agents (IF)", ["S_vel", "S_if", "S_gra"]),
                     ("Supervised Behavior", ["S_vel", "S_gb", "S_gra"])]:
    print(f"\n=== {label} ===")
    evaluate("Fixed paper weights", score(val, FIXED, feats), score(ev, FIXED, feats))
    L = learn(val, feats)
    print("  learned weights:", {c: np.round(w, 3).tolist() for c, w in L.items()})
    evaluate("Learned weights    ", score(val, L, feats), score(ev, L, feats))

print()
evaluate("S_gb alone (reference)", val["S_gb"].values, ev["S_gb"].values)