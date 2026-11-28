"""Reproduce the FraudLens Sparkov findings:  python scripts/run_evaluation.py"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudlens.agents.behavior import BehaviorAgent
from fraudlens.agents.graph_risk import GraphRiskAgent
from fraudlens.agents.velocity import VelocityAgent
from fraudlens.config import load_config
from fraudlens.features import build_features
from fraudlens.otp_interlock import OTPInterlock
from fraudlens.synthesis import Synthesis, norm_context

SETUPS = [("fixed_paper", "Fixed", "Paper"),
          ("fixed_cal", "Fixed", "Calibrated"),
          ("learned_cal", "Learned", "Calibrated"),
          ("behavior_alone", "Behavior alone", "Calibrated")]
NAMES = {"isolation_forest": "Isolation Forest", "supervised": "Supervised (extension)"}


def split(df, cfg):
    e = cfg["evaluation"]
    cut = df["ts"].iloc[int(len(df) * e["train_frac"])]
    train = df[df["ts"] < cut]
    test = df[df["ts"] >= cut].copy()
    mid = test["ts"].iloc[int(len(test) * e["calibration_frac_of_test"])]
    return train, test[test["ts"] < mid].copy(), test[test["ts"] >= mid].copy()


def metrics(y, tiers):
    esc = tiers > 0
    tp = int((esc & (y == 1)).sum())
    return {"precision": tp / max(int(esc.sum()), 1),
            "recall": tp / max(int((y == 1).sum()), 1),
            "fpr": int((esc & (y == 0)).sum()) / max(int((y == 0).sum()), 1)}


def evaluate(it, R, ctx, y):
    tiers = it.decide(R, ctx)
    cn = norm_context(ctx)
    res = {"roc_auc": float(roc_auc_score(y, R)),
           "pr_auc": float(average_precision_score(y, R)),
           "overall": metrics(y, tiers), "by_context": {}, "distribution": {},
           "thresholds": {str(c): it.thresholds[str(c)] for c in np.unique(cn)}}
    for c in np.unique(cn):
        m = cn == c
        res["by_context"][str(c)] = metrics(y[m], tiers[m])
    for cls, name in [(0, "legit"), (1, "fraud")]:
        t = tiers[y == cls]
        res["distribution"][name] = {a: float((t == i).mean() * 100)
                                     for i, a in enumerate(it.actions)}
    return res


def weights_dict(syn, ctxs, learned):
    W = syn.weights(learned)
    return {c: dict(zip(syn.agents, np.round(W[c], 3).tolist())) for c in ctxs}


def run_variant(cfg, sv, se, val, ev):
    yv, ye = val["is_fraud"].to_numpy(), ev["is_fraud"].to_numpy()
    cv, ce = val["context"].to_numpy(), ev["context"].to_numpy()
    ctxs = [str(c) for c in np.unique(norm_context(cv))]
    syn = Synthesis(cfg)
    Rv, Re = syn.composite(sv, cv), syn.composite(se, ce)
    out = {"fixed_paper": evaluate(OTPInterlock(cfg), Re, ce, ye)}
    out["fixed_cal"] = evaluate(OTPInterlock(cfg).calibrate(Rv, cv, yv), Re, ce, ye)
    out["behavior_alone"] = evaluate(OTPInterlock(cfg).calibrate(sv["behavior"], cv, yv), se["behavior"], ce, ye)
    out["weights_fixed"] = weights_dict(syn, ctxs, False)
    syn.fit_learned(sv, cv, yv)
    Rv, Re = syn.composite(sv, cv, True), syn.composite(se, ce, True)
    out["learned_cal"] = evaluate(OTPInterlock(cfg).calibrate(Rv, cv, yv), Re, ce, ye)
    out["weights_learned"] = weights_dict(syn, ctxs, True)
    return out


def to_md(res):
    L = ["#### Overall (second half of the test period)", "",
         "| Behavior | Weights | Thresholds | ROC-AUC | PR-AUC | Precision | Recall | FPR |",
         "|---|---|---|---|---|---|---|---|"]
    for v, runs in res["variants"].items():
        for key, w, t in SETUPS:
            r, o = runs[key], runs[key]["overall"]
            L.append(f"| {NAMES[v]} | {w} | {t} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} "
                     f"| {o['precision']:.3f} | {o['recall']:.3f} | {o['fpr']:.2%} |")
    L += ["", "#### By context", "",
          "| Behavior | Weights | Thresholds | Context | Precision | Recall | FPR |",
          "|---|---|---|---|---|---|---|"]
    for v, runs in res["variants"].items():
        for key, w, t in SETUPS:
            for c, m in runs[key]["by_context"].items():
                L.append(f"| {NAMES[v]} | {w} | {t} | {c.upper()} | {m['precision']:.3f} "
                         f"| {m['recall']:.3f} | {m['fpr']:.2%} |")
    L += ["", "#### Decision distribution (% of each class)", "",
          "| Behavior | Weights | Thresholds | Class | Approve | Single OTP | Dual OTP | Block |",
          "|---|---|---|---|---|---|---|---|"]
    for v, runs in res["variants"].items():
        for key, w, t in SETUPS:
            for cls, d in runs[key]["distribution"].items():
                L.append(f"| {NAMES[v]} | {w} | {t} | {cls} | {d['approve']:.1f} "
                         f"| {d['single_otp']:.1f} | {d['dual_otp']:.1f} | {d['block']:.1f} |")
    L += ["", "#### Weights (order: velocity, behavior, graph_risk)", "",
          "| Behavior | Type | Context | Weights |", "|---|---|---|---|"]
    for v, runs in res["variants"].items():
        for kind in ["fixed", "learned"]:
            for c, w in runs[f"weights_{kind}"].items():
                L.append(f"| {NAMES[v]} | {kind.title()} | {c.upper()} | "
                         f"{', '.join(f'{x:.3f}' for x in w.values())} |")
    ref = res["reference_supervised_alone"]
    L += ["", f"Supervised Behavior score alone: ROC-AUC {ref['roc_auc']:.3f}, "
              f"PR-AUC {ref['pr_auc']:.3f}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild features parquet")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = load_config(args.profile)
    if cfg["agents"]["geo"]["enabled"]:
        raise SystemExit("Geo needs the synthetic generator; use the sparkov profile")

    t0 = time.time()
    paths = cfg["evaluation"]["paths"]
    feats = Path(paths["features"])
    if args.rebuild or not feats.exists():
        print("Building features ...")
        df = build_features(cfg, paths["raw_csv"], save_to=feats)
    else:
        df = pd.read_parquet(feats)
    df = df.sort_values("ts").reset_index(drop=True)

    train, val, ev = split(df, cfg)
    print(f"train {len(train):,} | val {len(val):,} | eval {len(ev):,} "
          f"| eval fraud rate {ev['is_fraud'].mean():.3%}")

    velocity = VelocityAgent(cfg)
    graph = GraphRiskAgent(cfg).fit(train)
    print("Fitting behavior agent (per-card Isolation Forest + supervised) ...")
    behavior = BehaviorAgent(cfg).fit(train)

    def base(d):
        return {"velocity": velocity.score(d), "graph_risk": graph.score(d)}

    res = {"variants": {}}
    for name, fn in [("isolation_forest", behavior.score),
                     ("supervised", behavior.score_supervised)]:
        sv = {**base(val), "behavior": fn(val)}
        se = {**base(ev), "behavior": fn(ev)}
        res["variants"][name] = run_variant(cfg, sv, se, val, ev)

    s_gb = behavior.score_supervised(ev)
    y = ev["is_fraud"].to_numpy()
    res["reference_supervised_alone"] = {"roc_auc": float(roc_auc_score(y, s_gb)),
                                         "pr_auc": float(average_precision_score(y, s_gb))}
    res["meta"] = {"profile": cfg["profile"], "train_rows": len(train),
                   "val_rows": len(val), "eval_rows": len(ev),
                   "eval_fraud_rate": float(y.mean()),
                   "seconds": round(time.time() - t0, 1)}

    out = Path(paths["artifacts"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    md = to_md(res)
    (out / "results.md").write_text(md)
    print("\n" + md)
    print(f"\nDone in {res['meta']['seconds']} s. Saved {out / 'results.json'} and {out / 'results.md'}")


if __name__ == "__main__":
    main()