from copy import deepcopy
from importlib import resources
from pathlib import Path

import yaml

AGENTS = ["velocity", "geo", "behavior", "graph_risk"]


def _merge(base, over):
    out = deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(profile=None, path=None):
    if path:
        raw = yaml.safe_load(Path(path).read_text())
    else:
        raw = yaml.safe_load(
            resources.files("fraudlens").joinpath("config.yaml").read_text()
        )
    name = profile or raw["profile"]
    cfg = {k: v for k, v in raw.items() if k != "profiles"}
    if name != "sparkov":
        cfg = _merge(cfg, raw["profiles"][name])
    cfg["profile"] = name
    validate(cfg)
    return cfg


def validate(cfg):
    for ctx, w in cfg["synthesis"]["paper_weights"].items():
        assert set(w) == set(AGENTS), f"{ctx}: wrong agents"
        assert all(x >= 0 for x in w.values()), f"{ctx}: negative weight"
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"{ctx}: weights must sum to 1"
    for ctx, t in cfg["otp_interlock"]["paper_thresholds"].items():
        assert t == sorted(t) and len(set(t)) == 3, f"{ctx}: thresholds not increasing"
    for name, win in cfg["agents"]["velocity"]["windows"].items():
        assert 0 <= win["severity"] <= 1, f"{name}: severity outside [0,1]"