import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest


class BehaviorAgent:
    name = "behavior"

    def __init__(self, cfg):
        a = cfg["agents"]["behavior"]
        self.enabled = a["enabled"]
        self.beta = a["beta_cold_start"]
        self.min_rows = a["min_train_rows"]
        self.feats = a["features"]
        self.if_params = dict(a["isolation_forest"])
        self.sup = a["supervised"]
        self.gap_median = None
        self.categories = None
        self.cards = {}
        self.gb = None

    def _prepare(self, df):
        out = df[["log_amt", "hour_sin", "hour_cos", "vel_10m", "vel_1h"]].copy()
        out["log_gap"] = np.log1p(df["gap_sec"].fillna(self.gap_median))
        out["cat"] = pd.Categorical(df["category"], categories=self.categories).codes
        return out[self.feats]

    def fit(self, train):
        self.gap_median = float(train["gap_sec"].median())
        self.categories = sorted(train["category"].unique())
        X = self._prepare(train)
        legit = train["is_fraud"].to_numpy() == 0
        cards = train["cc_num"].to_numpy()[legit]
        self.cards = {}
        for card, g in X[legit].groupby(cards):
            if len(g) < self.min_rows:
                continue
            m = IsolationForest(**self.if_params).fit(g)
            self.cards[card] = (m, np.sort(-m.score_samples(g)))
        if self.sup["enabled"]:
            self.gb = HistGradientBoostingClassifier(
                categorical_features=[self.feats.index("cat")],
                max_iter=self.sup["max_iter"],
                learning_rate=self.sup["learning_rate"],
                random_state=self.sup["random_state"],
            ).fit(X, train["is_fraud"])
        return self

    def score(self, df):
        """Per-card Isolation Forest, rank-normalised to [0, 1]. beta if no history."""
        if not self.enabled:
            return np.zeros(len(df))
        out = np.full(len(df), self.beta)
        X = self._prepare(df)
        groups = pd.Series(np.arange(len(df))).groupby(df["cc_num"].to_numpy())
        for card, pos in groups.indices.items():
            if card not in self.cards:
                continue
            m, ref = self.cards[card]
            s = -m.score_samples(X.iloc[pos])
            out[pos] = np.searchsorted(ref, s) / len(ref)
        return out

    def score_supervised(self, df):
        if self.gb is None:
            raise RuntimeError("supervised model disabled or not fitted")
        return self.gb.predict_proba(self._prepare(df))[:, 1]