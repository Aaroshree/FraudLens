import pandas as pd

df = pd.read_csv("data/raw/fraudTrain.csv")
df["ts"] = pd.to_datetime(df["trans_date_trans_time"])

print("Date range:", df["ts"].min(), "to", df["ts"].max())
print("Cards:", df["cc_num"].nunique())
print("Merchants:", df["merchant"].nunique())
print("Txns per card (median):", df.groupby("cc_num").size().median())
print()
print("Fraud rate by category:")
print(df.groupby("category")["is_fraud"].agg(["count", "mean"]).sort_values("mean", ascending=False))
print()
print("Amount, fraud vs not:")
print(df.groupby("is_fraud")["amt"].describe())