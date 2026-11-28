import numpy as np
import pytest

from fraudlens.agents.geo import GeoAgent
from fraudlens.config import load_config
from fraudlens.otp_interlock import OTPInterlock
from fraudlens.synthesis import Synthesis


@pytest.mark.parametrize("profile", ["sparkov", "paper"])
def test_weights_sum_to_one_and_nonnegative(profile):
    s = Synthesis(load_config(profile))
    for w in s.weights().values():
        assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()


def test_sparkov_weights_match_findings():
    s = Synthesis(load_config("sparkov"))
    assert s.agents == ["velocity", "behavior", "graph_risk"]
    np.testing.assert_allclose(s.weights()["pos"], [0.5333, 0.2667, 0.2], atol=1e-3)
    np.testing.assert_allclose(s.weights()["qr"], [0.4, 0.3333, 0.2667], atol=1e-3)


def test_composite_in_unit_interval():
    s = Synthesis(load_config("paper"))
    rng = np.random.default_rng(0)
    n = 500
    scores = {a: rng.random(n) for a in s.agents}
    ctx = rng.choice(["remittance", "POS", "QR"], n)
    R = s.composite(scores, ctx)
    assert ((R >= 0) & (R <= 1)).all()


def test_sita_example_is_dual_otp():
    cfg = load_config("paper")
    geo = GeoAgent.combine(0.92, 1.0, cfg["agents"]["geo"]["alpha"])
    scores = {"velocity": [0.05], "geo": geo, "behavior": [0.88], "graph_risk": [0.97]}
    R = Synthesis(cfg).composite(scores, ["remittance"])[0]
    assert abs(R - 0.797) < 0.005
    it = OTPInterlock(cfg)
    assert it.labels(it.decide([R], ["remittance"]))[0] == "dual_otp"


def test_paper_threshold_boundaries():
    it = OTPInterlock(load_config("paper"))
    R = [0.349, 0.35, 0.599, 0.60, 0.799, 0.80]
    assert list(it.decide(R, ["remittance"] * 6)) == [0, 1, 1, 2, 2, 3]


def test_calibration_hits_target_fpr():
    it = OTPInterlock(load_config("sparkov"))
    rng = np.random.default_rng(1)
    n = 20000
    R = rng.random(n)
    y = np.zeros(n, int)
    y[:200] = 1
    ctx = np.array(["POS"] * n)
    it.calibrate(R, ctx, y)
    tiers = it.decide(R, ctx)
    assert abs((tiers[y == 0] > 0).mean() - 0.05) < 0.005


def test_learned_weights_valid_and_behavior_dominates():
    s = Synthesis(load_config("sparkov"))
    rng = np.random.default_rng(2)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    scores = {
        "velocity": rng.random(n) * 0.5 + y * 0.05,
        "behavior": rng.random(n) * 0.5 + y * 0.4,
        "graph_risk": rng.random(n),
    }
    ctx = rng.choice(["POS", "QR"], n)
    s.fit_learned(scores, ctx, y)
    for w in s.weights(learned=True).values():
        assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    assert s.weights(learned=True)["pos"][s.agents.index("behavior")] > 0.5