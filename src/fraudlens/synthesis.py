import numpy as np
from sklearn.linear_model import LogisticRegression

ORDER = ["velocity", "geo", "behavior", "graph_risk"]


def norm_context(context):
    return np.char.lower(np.asarray(context, dtype=str))


class Synthesis:
    def __init__(self, cfg):
        self.cfg = cfg
        s = cfg["synthesis"]
        self.agents = [a for a in ORDER if cfg["agents"][a]["enabled"]]
        if len(self.agents) < len(ORDER) and not s["renormalise_disabled"]:
            raise ValueError("agents disabled but renormalise_disabled is false")
        self.fixed = {}
        for ctx, w in s["paper_weights"].items():
            v = np.array([w[a] for a in self.agents], float)
            self.fixed[ctx] = v / v.sum()
        self.learned = {}

    def weights(self, learned=False):
        return self.learned if learned else self.fixed

    def _matrix(self, scores):
        return np.column_stack(
            [np.atleast_1d(np.asarray(scores[a], float)) for a in self.agents]
        )

    def composite(self, scores, context, learned=False):
        X = self._matrix(scores)
        ctx = norm_context(context)
        W = self.weights(learned)
        R = np.zeros(len(X))
        for c in np.unique(ctx):
            m = ctx == c
            R[m] = X[m] @ W[c]
        return R

    def fit_learned(self, scores, context, y):
        p = self.cfg["synthesis"]["learned"]
        X = self._matrix(scores)
        ctx = norm_context(context)
        y = np.asarray(y)
        self.learned = {c: w.copy() for c, w in self.fixed.items()}
        for c in np.unique(ctx):
            m = ctx == c
            if y[m].sum() == 0:
                continue
            lr = LogisticRegression(
                class_weight=p["class_weight"], C=p["C"], max_iter=p["max_iter"]
            ).fit(X[m], y[m])
            w = np.clip(lr.coef_[0], 0, None)
            if w.sum() > 0:
                self.learned[c] = w / w.sum()
        return self