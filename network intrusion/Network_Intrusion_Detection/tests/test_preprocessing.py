import numpy as np
import pandas as pd
import pytest

from models.preprocessing import RAW_FEATURES, fit_preprocessor, transform_features


def sample_frame(rows=40):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({name: rng.uniform(1, 100, rows) for name in RAW_FEATURES})
    frame["Destination Port"] = rng.choice([22, 80, 443], rows)
    frame["Flow Duration"] = rng.uniform(1000, 2_000_000, rows)
    frame["Label"] = np.where(np.arange(rows) % 2 == 0, "Benign", "DDoS")
    return frame


def test_train_and_inference_features_match():
    frame = sample_frame()
    train_x, state = fit_preprocessor(frame.iloc[:30])
    test_x = transform_features(frame.iloc[30:], state)
    assert train_x.columns.tolist() == test_x.columns.tolist()
    assert np.isfinite(train_x.to_numpy()).all()
    assert np.isfinite(test_x.to_numpy()).all()


def test_missing_required_raw_feature_fails_loudly():
    frame = sample_frame().drop(columns=["Flow Duration"])
    with pytest.raises(ValueError, match="missing required CICIDS2017 columns"):
        fit_preprocessor(frame)


def test_test_only_destination_port_does_not_change_training_mapping():
    train = sample_frame(30)
    _, state = fit_preprocessor(train)
    unseen = sample_frame(3).drop(columns=["Label"])
    unseen["Destination Port"] = 65000
    transformed = transform_features(unseen, state)
    assert (transformed["destination_port_frequency"] == 0).all()
