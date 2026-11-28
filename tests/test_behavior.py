import numpy as np
import pandas as pd

from fraudlens.agents.behavior import BehaviorAgent
from fraudlens.config import load_config


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.integers(8, 20, n)
    df = pd.DataFrame({
        "cc_num": np.repeat([1, 2], n // 2),
        "category": rng.choice(["food", "gas_transport"], n),
        "is_fraud": 0,
        "log_amt": rng.normal(3.0, 0.2, n),
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "vel_10m": 0,
        "vel_1h": rng.integers(0, 2, n),
        "gap_sec": rng.integers(3600, 86400, n).astype(float),
    })
    fraud = rng.choice(n, 20, replace=False)
    df.loc[fraud, "is_fraud"] = 1
    df.loc[fraud, "log_amt"] = 6.0
    return df


def _probe(df):
    base = df[(df.cc_num == 1) & (df.is_fraud == 0)].iloc[[0]]
    bad = base.copy()
    bad["log_amt"] = 9.0
    return pd.concat([base, bad], ignore_index=True)


def test_unseen_card_gets_beta():
    df = _toy()
    agent = BehaviorAgent(load_config("sparkov")).fit(df)
    new = df.iloc[[0]].copy()
    new["cc_num"] = 999
    assert agent.score(new)[0] == 0.5


def test_if_scores_in_unit_interval_and_anomaly_higher():
    df = _toy()
    agent = BehaviorAgent(load_config("sparkov")).fit(df)
    s = agent.score(_probe(df))
    assert ((s >= 0) & (s <= 1)).all()
    assert s[1] > s[0]


def test_supervised_in_unit_interval_and_fraud_like_higher():
    df = _toy()
    agent = BehaviorAgent(load_config("sparkov")).fit(df)
    s = agent.score_supervised(_probe(df))
    assert ((s >= 0) & (s <= 1)).all()
    assert s[1] > s[0]