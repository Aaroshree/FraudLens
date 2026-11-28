import json
from pathlib import Path

import numpy as np

from fraudlens.synthesis import norm_context


class OTPInterlock:
    def __init__(self, cfg):
        c = cfg["otp_interlock"]
        self.actions = c["actions"]
        self.thresholds = {k: list(v) for k, v in c["paper_thresholds"].items()}
        self.pcts = c["calibration"]["percentiles"]

    def calibrate(self, R, context, y):
        """Set thresholds to percentiles of legit R per context."""
        R, y = np.asarray(R, float), np.asarray(y)
        ctx = norm_context(context)
        q = np.array(self.pcts) / 100.0
        for c in np.unique(ctx):
            m = (ctx == c) & (y == 0)
            if m.any():
                self.thresholds[c] = np.quantile(R[m], q).tolist()
        return self

    def decide(self, R, context):
        R = np.asarray(R, float)
        ctx = norm_context(context)
        tiers = np.zeros(len(R), dtype=int)
        for c in np.unique(ctx):
            m = ctx == c
            tiers[m] = np.digitize(R[m], self.thresholds[c])
        return tiers

    def labels(self, tiers):
        return np.array(self.actions)[np.asarray(tiers, dtype=int)]

    def save(self, path):
        Path(path).write_text(json.dumps(self.thresholds, indent=2))

    def load(self, path):
        self.thresholds = json.loads(Path(path).read_text())
        return self