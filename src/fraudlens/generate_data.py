"""Generate a synthetic Nepal digital payments dataset for FraudLens.

This is a NEW generator inspired by the paper, not the original script.
Dataset size, transaction mix and fraud-pattern shares follow the paper's
Tables 7 to 9. Everything else (amounts, cities, devices) is invented here.

Run from the project root:
    python src/fraudlens/generate_data.py

Output: data/raw/transactions.csv
The seed is fixed, so the file is identical on every run.
All values are simulated. Nothing here is real banking data.

Model inputs must NEVER include: is_fraud, fraud_pattern.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- Settings ----------
SEED = 42
N_TOTAL = 8500
N_USERS = 200
N_NEW_USERS = 60          # users who join during the window (cold-start)
FRAUD_RATE = 0.053
WINDOW_DAYS = 90
START = pd.Timestamp("2026-01-01")

TYPES = ["remittance", "POS", "QR"]
TYPE_P = [0.20, 0.45, 0.35]

# (median amount in NPR, spread) per transaction type
AMOUNT = {"remittance": (20000, 0.30), "POS": (1500, 0.60), "QR": (400, 0.60)}

FRAUD_SHARES = {
    "velocity_burst": 0.25,
    "location_anomaly": 0.20,
    "new_merchant": 0.20,
    "sim_swap": 0.15,
    "cold_start": 0.10,
    "amount_spike": 0.10,
}

CITIES = {  # name: (latitude, longitude)
    "Kathmandu": (27.70, 85.32), "Lalitpur": (27.67, 85.32),
    "Bhaktapur": (27.67, 85.43), "Pokhara": (28.21, 83.99),
    "Biratnagar": (26.45, 87.27), "Butwal": (27.70, 83.45),
    "Birgunj": (27.01, 84.88), "Dharan": (26.81, 87.28),
    "Nepalgunj": (28.05, 81.62), "Janakpur": (26.73, 85.93),
}
CITY_P = [0.35, 0.08, 0.05, 0.10, 0.08, 0.08, 0.06, 0.05, 0.05, 0.10]
CATEGORIES = ["retail", "food", "utility", "transport", "education", "health"]

COLUMNS = [
    "transaction_id", "user_id", "timestamp", "txn_type", "amount",
    "merchant_id", "merchant_category", "city", "latitude", "longitude",
    "device_id", "is_fraud", "fraud_pattern",
]


def fraud_counts():
    """How many fraud rows each pattern should have."""
    n_fraud = round(N_TOTAL * FRAUD_RATE)
    counts = {k: int(n_fraud * s) for k, s in FRAUD_SHARES.items()}
    counts["velocity_burst"] += n_fraud - sum(counts.values())  # rounding leftovers
    return counts


# ---------- Helpers ----------
def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, [*CITIES[a], *CITIES[b]])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def new_device_id(rng):
    """Opaque device id. Every device, real or fraudulent, looks the same."""
    return f"D{int(rng.integers(0, 16 ** 8)):08x}"


def build_world(rng):
    """98 merchants with opaque ids: 8 remittance corridors, 30 rare, 60 common."""
    ids = [f"M{i:03d}" for i in range(1, 99)]
    order = rng.permutation(98)
    remit = [ids[i] for i in order[:8]]
    rare = [ids[i] for i in order[8:38]]
    common = [ids[i] for i in order[38:]]
    shops = common + rare
    merch_cat = {m: str(rng.choice(CATEGORIES)) for m in shops}
    merch_cat.update({m: "transfer" for m in remit})
    pop = rng.lognormal(0, 1.0, len(shops))
    pop[len(common):] *= 0.15            # rare merchants are rarely used by honest users
    return {"common": common, "rare": rare, "shops": shops,
            "pop_p": pop / pop.sum(), "remit": remit, "merch_cat": merch_cat}


def build_users(rng, w):
    is_new = np.zeros(N_USERS, dtype=bool)
    is_new[rng.choice(N_USERS, N_NEW_USERS, replace=False)] = True
    users = []
    for i in range(N_USERS):
        users.append({
            "user_id": f"U{i + 1:04d}",
            "home_city": str(rng.choice(list(CITIES), p=CITY_P)),
            "scale": float(rng.lognormal(0, 0.2)),
            "peak_hour": float(rng.uniform(10, 20)),
            "start_day": float(rng.uniform(20, 75)) if is_new[i] else 0.0,
            "activity": float(rng.lognormal(0, 0.5)) * (0.5 if is_new[i] else 1.0),
            "favorites": [str(m) for m in rng.choice(w["common"], 6, replace=False)],
            "device": new_device_id(rng),
            "device_b": new_device_id(rng),   # a second phone or tablet
        })
    return users


def pick_type(rng):
    return str(rng.choice(TYPES, p=TYPE_P))


def draw_amount(rng, u, t):
    median, sigma = AMOUNT[t]
    return rng.lognormal(np.log(median * u["scale"]), sigma)


def typical_max(u, t):
    median, sigma = AMOUNT[t]
    return median * u["scale"] * np.exp(1.5 * sigma)


def draw_ts(rng, u, hour):
    day = min(int(rng.uniform(u["start_day"], WINDOW_DAYS)), WINDOW_DAYS - 1)
    return START + pd.Timedelta(days=day) + pd.Timedelta(hours=float(hour % 24))


def any_merchant(rng, w):
    return str(rng.choice(w["shops"], p=w["pop_p"]))


def make_row(rng, w, u, ts, t, amount, merchant, city, device, pattern):
    lat, lon = CITIES[city]
    return {
        "user_id": u["user_id"], "timestamp": ts, "txn_type": t,
        "amount": round(float(amount), 2), "merchant_id": merchant,
        "merchant_category": w["merch_cat"][merchant], "city": city,
        "latitude": round(lat + rng.normal(0, 0.01), 4),
        "longitude": round(lon + rng.normal(0, 0.01), 4),
        "device_id": device, "is_fraud": int(pattern != "none"),
        "fraud_pattern": pattern,
    }


# ---------- Legitimate transactions ----------
def gen_legit(rng, w, users, n):
    p = np.array([u["activity"] for u in users])
    p = p / p.sum()
    rows = []
    for idx in rng.choice(len(users), size=n, p=p):
        u = users[idx]
        t = pick_type(rng)
        hour = rng.normal(20, 1.5) if t == "remittance" else rng.normal(u["peak_hour"], 2.5)
        if t == "remittance":
            merchant = str(rng.choice(w["remit"]))
        elif rng.random() < 0.8:
            merchant = str(rng.choice(u["favorites"]))
        else:
            merchant = any_merchant(rng, w)
        city = u["home_city"] if rng.random() < 0.95 else str(rng.choice(list(CITIES)))
        device = u["device"] if rng.random() < 0.97 else u["device_b"]
        rows.append(make_row(rng, w, u, draw_ts(rng, u, hour), t,
                             draw_amount(rng, u, t), merchant, city, device, "none"))
    return pd.DataFrame(rows)


# ---------- Fraud patterns ----------
def gen_velocity(rng, w, users, n):
    est = [u for u in users if u["start_day"] == 0]
    n_bursts = n // 6
    sizes = [6] * n_bursts
    for i in range(n - 6 * n_bursts):       # spread the remainder, every burst >= 5
        sizes[i] += 1
    rows = []
    for size in sizes:
        u = est[rng.integers(len(est))]
        t = str(rng.choice(["POS", "QR"]))
        base = draw_ts(rng, u, rng.uniform(8, 22))
        merchant = any_merchant(rng, w)
        for i in range(size):                # all inside one minute
            rows.append(make_row(rng, w, u, base + pd.Timedelta(seconds=8 * i), t,
                                 draw_amount(rng, u, t), merchant, u["home_city"],
                                 u["device"], "velocity_burst"))
    return rows


def gen_location(rng, w, users, legit, n):
    by_id = {u["user_id"]: u for u in users}
    rows = []
    for i in rng.choice(len(legit), size=n, replace=False):
        base = legit.iloc[i]
        u = by_id[base["user_id"]]
        far = [c for c in CITIES if haversine_km(u["home_city"], c) > 250]
        t = str(rng.choice(TYPES, p=[0.5, 0.3, 0.2]))
        merchant = str(rng.choice(w["remit"])) if t == "remittance" else any_merchant(rng, w)
        rows.append(make_row(rng, w, u, base["timestamp"] + pd.Timedelta(minutes=10), t,
                             draw_amount(rng, u, t), merchant, str(rng.choice(far)),
                             u["device"], "location_anomaly"))
    return rows


def gen_new_merchant(rng, w, users, n):
    rows = []
    for _ in range(n):
        u = users[rng.integers(len(users))]
        t = str(rng.choice(["POS", "QR"], p=[0.4, 0.6]))
        rows.append(make_row(rng, w, u, draw_ts(rng, u, rng.uniform(8, 22)), t,
                             draw_amount(rng, u, t), str(rng.choice(w["rare"])),
                             u["home_city"], u["device"], "new_merchant"))
    return rows


def gen_sim_swap(rng, w, users, n):
    rows = []
    for _ in range(n):
        u = users[rng.integers(len(users))]
        t = str(rng.choice(TYPES, p=[0.7, 0.15, 0.15]))
        city = u["home_city"] if rng.random() < 0.5 else str(rng.choice(list(CITIES)))
        amount = typical_max(u, t) * rng.uniform(1.5, 3.0)
        rows.append(make_row(rng, w, u, draw_ts(rng, u, rng.uniform(1, 4)), t, amount,
                             str(rng.choice(w["rare"])), city,
                             new_device_id(rng), "sim_swap"))   # a phone never seen before
    return rows


def gen_cold_start(rng, w, users, legit, n):
    times = legit.sort_values("timestamp").groupby("user_id")["timestamp"].apply(list)
    eligible = [u for u in users if u["start_day"] > 0 and len(times.get(u["user_id"], [])) >= 4]
    rows = []
    for i in rng.choice(len(eligible), size=n, replace=False):
        u = eligible[i]
        ts_list = times[u["user_id"]]
        position = int(rng.integers(1, 4))   # becomes the user's 1st, 2nd or 3rd transaction
        if position == 1:
            ts = ts_list[0] - pd.Timedelta(minutes=30)
        else:
            ts = ts_list[position - 2] + (ts_list[position - 1] - ts_list[position - 2]) / 2
        t = pick_type(rng)
        amount = typical_max(u, t) * rng.uniform(1.0, 2.0)
        rows.append(make_row(rng, w, u, ts, t, amount, str(rng.choice(w["rare"])),
                             u["home_city"], u["device"], "cold_start"))
    return rows


def gen_amount_spike(rng, w, users, n):
    rows = []
    for _ in range(n):
        u = users[rng.integers(len(users))]
        t = pick_type(rng)
        merchant = str(rng.choice(w["remit"])) if t == "remittance" else str(rng.choice(u["favorites"]))
        amount = typical_max(u, t) * rng.uniform(2.5, 6.0)
        rows.append(make_row(rng, w, u, draw_ts(rng, u, rng.uniform(0, 24)), t, amount,
                             merchant, u["home_city"], u["device"], "amount_spike"))
    return rows


# ---------- Build the dataset ----------
def generate():
    rng = np.random.default_rng(SEED)
    w = build_world(rng)
    users = build_users(rng, w)
    counts = fraud_counts()
    n_fraud = sum(counts.values())

    legit = gen_legit(rng, w, users, N_TOTAL - n_fraud)

    # Cold-start fraud goes first. Those users get no other fraud, so the
    # cold-start transaction really is among their first three.
    cold_rows = gen_cold_start(rng, w, users, legit, counts["cold_start"])
    cold_ids = {r["user_id"] for r in cold_rows}
    free_users = [u for u in users if u["user_id"] not in cold_ids]
    free_legit = legit[~legit["user_id"].isin(cold_ids)].reset_index(drop=True)

    fraud_rows = (
        cold_rows
        + gen_velocity(rng, w, users, counts["velocity_burst"])
        + gen_location(rng, w, users, free_legit, counts["location_anomaly"])
        + gen_new_merchant(rng, w, free_users, counts["new_merchant"])
        + gen_sim_swap(rng, w, free_users, counts["sim_swap"])
        + gen_amount_spike(rng, w, free_users, counts["amount_spike"])
    )
    df = pd.concat([legit, pd.DataFrame(fraud_rows)], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i + 1:06d}" for i in range(len(df))])
    return df[COLUMNS]


# ---------- Validation ----------
def validate(df):
    """Return a list of problems. An empty list means every check passed."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]

    errors = []
    if len(df) != N_TOTAL:
        errors.append(f"Expected {N_TOTAL} rows, found {len(df)}")
    if df["user_id"].nunique() != N_USERS:
        errors.append(f"Expected {N_USERS} users, found {df['user_id'].nunique()}")
    if not df["transaction_id"].is_unique:
        errors.append("transaction_id is not unique")
    if df.isna().any().any():
        errors.append("Dataset contains missing values")
    if (df["amount"] <= 0).any():
        errors.append("Found amounts that are zero or negative")
    if not df["txn_type"].isin(TYPES).all():
        errors.append("Unknown txn_type value")
    if not df["city"].isin(CITIES).all():
        errors.append("Unknown city value")
    if not df["is_fraud"].isin([0, 1]).all():
        errors.append("is_fraud must be 0 or 1")
    if not df["latitude"].between(25, 31).all() or not df["longitude"].between(79, 89).all():
        errors.append("Coordinates fall outside Nepal")
    ts = pd.to_datetime(df["timestamp"])
    if ts.min() < START or ts.max() >= START + pd.Timedelta(days=WINDOW_DAYS):
        errors.append("Timestamps fall outside the 90-day window")

    # Labels and patterns must agree, and counts must match the configuration
    if ((df["is_fraud"] == 1) != (df["fraud_pattern"] != "none")).any():
        errors.append("is_fraud and fraud_pattern disagree")
    actual = df.loc[df["is_fraud"] == 1, "fraud_pattern"].value_counts().to_dict()
    if actual != fraud_counts():
        errors.append(f"Fraud pattern counts {actual} differ from config {fraud_counts()}")

    # IDs must not reveal the label through their naming format
    if not df["device_id"].map(lambda x: bool(re.fullmatch(r"D[0-9a-f]{8}", str(x)))).all():
        errors.append("device_id format is not uniform")
    if not df["merchant_id"].map(lambda x: bool(re.fullmatch(r"M\d{3}", str(x)))).all():
        errors.append("merchant_id format is not uniform")

    # Velocity bursts: every burst has 5+ transactions inside one minute
    v = df[df["fraud_pattern"] == "velocity_burst"].assign(ts=ts).sort_values(["user_id", "ts"])
    if len(v):
        gap = v.groupby("user_id")["ts"].diff()
        burst = (gap.isna() | (gap > pd.Timedelta(seconds=60))).cumsum()
        g = v.groupby(burst)["ts"].agg(["count", "min", "max"])
        if (g["count"] < 5).any() or ((g["max"] - g["min"]) >= pd.Timedelta(seconds=60)).any():
            errors.append("A velocity burst has fewer than 5 transactions or lasts over 60 seconds")

    # Cold-start fraud must be among the user's first 3 transactions
    ordered = df.assign(ts=ts).sort_values("ts")
    rank = ordered.groupby("user_id").cumcount() + 1
    cold = ordered["fraud_pattern"] == "cold_start"
    if (rank[cold] > 3).any():
        errors.append("A cold_start fraud is not within the user's first 3 transactions")
    return errors


def main():
    df = generate()
    errors = validate(df)
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))

    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "transactions.csv", index=False)

    print(f"Saved {len(df)} rows to {out_dir / 'transactions.csv'}")
    print("All validation checks passed")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.1%})")
    print("\nTransaction types:")
    print(df["txn_type"].value_counts(normalize=True).round(3).to_string())
    print("\nFraud patterns:")
    print(df.loc[df["is_fraud"] == 1, "fraud_pattern"].value_counts().to_string())


if __name__ == "__main__":
    main()
