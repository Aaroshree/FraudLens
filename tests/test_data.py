import pytest

from fraudlens.generate_data import generate, validate


@pytest.fixture(scope="module")
def df():
    return generate()


def test_generated_data_passes_all_checks(df):
    assert validate(df) == []


def test_same_seed_gives_same_data(df):
    assert generate().equals(df)


def test_negative_amount_is_caught(df):
    bad = df.copy()
    bad.loc[0, "amount"] = -5
    assert validate(bad) != []


def test_label_mismatch_is_caught(df):
    bad = df.copy()
    bad.loc[bad["is_fraud"] == 1, "is_fraud"] = 0
    assert validate(bad) != []


def test_duplicate_transaction_id_is_caught(df):
    bad = df.copy()
    bad.loc[1, "transaction_id"] = bad.loc[0, "transaction_id"]
    assert validate(bad) != []
