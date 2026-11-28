import numpy as np
import pandas as pd

from fraudlens.agents.graph_risk import GraphRiskAgent
from fraudlens.agents.velocity import VelocityAgent
from fraudlens.config import load_config


def _df():
    return pd.DataFrame({
        "vel_10m": [0, 2, 2, 5],
        "vel_1h": [0, 1, 3, 9],
        "vel_24h": [0, 0, 0, 0],
        "is_cold_start": [False] * 4,
        "um_prior": [0, 0, 1, 10],
        "m_prior_txns": [0, 5, 50, 100],
        "m_prior_users": [0, 1, 10, 20],
    })


def test_velocity_in_unit_interval_and_monotone():
    s = VelocityAgent(load_config("sparkov")).score(_df())
    assert ((s >= 0) & (s <= 1)).all()
    assert list(s) == [0.0, 0.4, 0.8, 0.8]


def test_velocity_amplification_capped_at_one():
    cfg = load_config("paper")
    cfg["agents"]["velocity"]["lambda_amp"] = 0.9
    df = pd.DataFrame({"vel_10m": [9], "vel_1h": [9], "vel_24h": [9],
                       "is_cold_start": [False]})
    assert VelocityAgent(cfg).score(df)[0] == 1.0


def test_graph_novelty_first_time_is_one():
    s = GraphRiskAgent(load_config("sparkov")).score(_df())
    assert s[0] == s[1] == 1.0
    assert s[1] > s[2] > s[3]
    assert ((s >= 0) & (s <= 1)).all()


def test_graph_paper_example_sita():
    # N = 1.0, F = 0.94, C = 0.97 -> 0.97
    assert abs(GraphRiskAgent.combine(1.0, 0.94, 0.97)[0] - 0.97) < 1e-9


def test_graph_full_profile_in_unit_interval():
    agent = GraphRiskAgent(load_config("paper")).fit(_df())
    s = agent.score(_df())
    assert ((s >= 0) & (s <= 1)).all()