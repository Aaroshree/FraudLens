import pandas as pd

from fraudlens.config import load_config
from fraudlens.features import build_features


def test_features_are_leak_free_counts(tmp_path):
    rows = [
        (1, "2020-01-01 10:00:00", "m1", "food_dining", 10, 0),
        (1, "2020-01-01 10:00:30", "m1", "shopping_net", 20, 0),
        (2, "2020-01-01 10:01:00", "m1", "food_dining", 5, 1),
        (1, "2020-01-01 12:00:00", "m2", "food_dining", 15, 0),
    ]
    p = tmp_path / "raw.csv"
    cols = ["cc_num", "trans_date_trans_time", "merchant", "category", "amt", "is_fraud"]
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    f = build_features(load_config(), p)
    assert list(f["vel_10m"]) == [0, 1, 0, 0]
    assert list(f["vel_24h"]) == [0, 1, 0, 2]
    assert list(f["um_prior"]) == [0, 1, 0, 0]
    assert list(f["m_prior_txns"]) == [0, 1, 2, 0]
    assert list(f["m_prior_users"]) == [0, 1, 1, 0]
    assert list(f["context"]) == ["POS", "QR", "POS", "POS"]