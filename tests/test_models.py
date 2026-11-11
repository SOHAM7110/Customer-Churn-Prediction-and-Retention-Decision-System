import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.features import FEATURES
from src.models import resample, fit_model, DEFAULTS

def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2, 20, size=(n, len(FEATURES))), columns=FEATURES)
    X["order_trend"] = rng.uniform(-1, 1, n); X["spend_trend"] = rng.uniform(-1, 1, n); X["is_uk"] = rng.integers(0, 2, n)
    y = (X.recency_days + rng.normal(0, 15, n) > 55).astype(int).values
    X.loc[rng.random(n) < 0.2, ["std_gap_days", "gap_cv"]] = np.nan          # missing values must not break LR
    return X, y

def test_resample_balances_classes():
    X, y = toy()
    for s in ("undersample", "oversample"):
        Xr, yr = resample(X, y, s)
        assert abs(yr.mean() - 0.5) < 1e-9 and len(Xr) == len(yr)
    Xn, yn = resample(X, y, "none"); assert len(Xn) == len(X)

def test_oversample_keeps_every_original_row():
    X, y = toy()
    Xr, _ = resample(X, y, "oversample")
    assert set(Xr.index) == set(X.index)

def test_lr_handles_nan_and_returns_probabilities():
    X, y = toy()
    for strat in ("none", "class_weight", "undersample", "oversample"):
        p = fit_model("lr", DEFAULTS["lr"], strat, X, y).predict_proba(X)[:, 1]
        assert p.min() >= 0 and p.max() <= 1 and not np.isnan(p).any()
        assert np.corrcoef(p, X.recency_days)[0, 1] > 0.3

def test_class_weight_raises_mean_probability():
    X, y = toy()
    p0 = fit_model("lr", DEFAULTS["lr"], "none", X, y).predict_proba(X)[:, 1].mean()
    p1 = fit_model("lr", DEFAULTS["lr"], "class_weight", X, y).predict_proba(X)[:, 1].mean()
    assert p1 > p0 if y.mean() < 0.5 else p1 < p0     # the reason Stage 5 calibration exists
