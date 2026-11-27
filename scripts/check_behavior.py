import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_parquet("data/processed/features.parquet")
df = df.sort_values("ts").reset_index(drop=True)
df["log_gap"] = np.log1p(df["gap_sec"].fillna(df["gap_sec"].median()))
df["cat"] = df["category"].astype("category").cat.codes

FEATS = ["log_amt", "hour_sin", "hour_cos", "cat", "vel_10m", "vel_1h", "log_gap"]

# Time-based split: first 70% of rows (by time) = train, last 30% = test
cut = df["ts"].iloc[int(len(df) * 0.7)]
train = df[df["ts"] < cut]
test = df[df["ts"] >= cut].copy()
print("Train:", len(train), "Test:", len(test), "| Test fraud rate:",
      round(test["is_fraud"].mean(), 4))

# ---------- Variant A: per-card Isolation Forest (paper), legit rows only ----------
test["S_if"] = 0.5  # beta = 0.5 default for cards with no history
test_groups = test.groupby("cc_num").groups
for card, tr in train[train["is_fraud"] == 0].groupby("cc_num"):
    idx = test_groups.get(card)
    if idx is None:
        continue
    m = IsolationForest(n_estimators=100, random_state=42).fit(tr[FEATS])
    tr_scores = np.sort(-m.score_samples(tr[FEATS]))
    s = -m.score_samples(test.loc[idx, FEATS])
    # score = share of the card's own train rows that look more normal (range 0 to 1)
    test.loc[idx, "S_if"] = np.searchsorted(tr_scores, s) / len(tr_scores)

# ---------- Variant B: supervised gradient boosting ----------
gb = HistGradientBoostingClassifier(
    categorical_features=[FEATS.index("cat")], random_state=42
)
gb.fit(train[FEATS], train["is_fraud"])
test["S_gb"] = gb.predict_proba(test[FEATS])[:, 1]

# ---------- Results ----------
def report(name, col):
    y = test["is_fraud"]
    print(f"\n{name}: ROC-AUC {roc_auc_score(y, test[col]):.4f} | "
          f"PR-AUC {average_precision_score(y, test[col]):.4f}")
    for c, d in test.groupby("context"):
        print(f"   {c}: ROC-AUC {roc_auc_score(d['is_fraud'], d[col]):.4f} | "
              f"PR-AUC {average_precision_score(d['is_fraud'], d[col]):.4f}")

report("Isolation Forest (per card)", "S_if")
report("Gradient boosting (supervised)", "S_gb")

test[["ts", "cc_num", "context", "is_fraud", "S_if", "S_gb"]].to_parquet(
    "data/processed/behavior_scores_test.parquet", index=False
)