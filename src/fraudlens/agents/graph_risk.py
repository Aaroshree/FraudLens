import numpy as np


class GraphRiskAgent:
    name = "graph_risk"

    def __init__(self, cfg):
        a = cfg["agents"]["graph_risk"]
        self.enabled = a["enabled"]
        self.k = a["k"]
        self.use = a["use"]
        self.t_max = None
        self.u_max = None

    def fit(self, df):
        """Learn merchant maxima from training rows only (leak-free)."""
        self.t_max = max(float(df["m_prior_txns"].max()), 1.0)
        self.u_max = max(float(df["m_prior_users"].max()), 1.0)
        return self

    @staticmethod
    def novelty(c, k):
        c = np.asarray(c, dtype=float)
        return np.where(c <= 0, 1.0, np.exp(-k * c))

    @staticmethod
    def combine(*components):
        return np.mean(np.vstack([np.asarray(c, float) for c in components]), axis=0)

    def components(self, df):
        out = {}
        if self.use["novelty"]:
            out["N"] = self.novelty(df["um_prior"].to_numpy(), self.k)
        if self.use["frequency"] or self.use["connectivity"]:
            if self.t_max is None:
                raise RuntimeError("call fit(train_df) before scoring F or C")
        if self.use["frequency"]:
            out["F"] = 1.0 - df["m_prior_txns"].to_numpy() / self.t_max
        if self.use["connectivity"]:
            out["C"] = 1.0 - df["m_prior_users"].to_numpy() / self.u_max
        return {k: np.clip(v, 0.0, 1.0) for k, v in out.items()}

    def score(self, df):
        if not self.enabled:
            return np.zeros(len(df))
        comps = self.components(df)
        if not comps:
            return np.zeros(len(df))
        return np.clip(self.combine(*comps.values()), 0.0, 1.0)