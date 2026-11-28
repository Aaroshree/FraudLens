import numpy as np


class VelocityAgent:
    name = "velocity"

    def __init__(self, cfg):
        a = cfg["agents"]["velocity"]
        self.enabled = a["enabled"]
        self.lam = a["lambda_amp"]
        self.gamma = a["cold_start_multiplier"]
        self.windows = {k: w for k, w in a["windows"].items() if w["enabled"]}

    def score(self, df):
        n = len(df)
        if not self.enabled or not self.windows:
            return np.zeros(n)
        if "is_cold_start" in df:
            cold = df["is_cold_start"].to_numpy(bool)
        else:
            cold = np.zeros(n, bool)
        scale = np.where(cold, self.gamma, 1.0)
        total = np.zeros(n)
        all_flags = np.ones(n, bool)
        for w in self.windows.values():
            flag = df[w["feature"]].to_numpy() > w["threshold"] * scale
            total += w["severity"] * flag
            all_flags &= flag
        total += self.lam * all_flags
        return np.clip(total, 0.0, 1.0)