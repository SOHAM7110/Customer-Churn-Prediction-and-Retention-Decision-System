import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.calibrate import Calibrator, calib_metrics, ece

def distorted(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.beta(2, 3, n); y = (rng.random(n) < true).astype(int)
    raw = np.clip(true ** 0.45, 0, 1)                       # systematically over-confident (inflated) scores
    return raw, y

def test_calibration_fixes_inflated_probabilities():
    raw, y = distorted()
    h = len(y) // 2
    base = calib_metrics(y[h:], raw[h:])
    for method in ("sigmoid", "isotonic"):
        p = Calibrator(method).fit(raw[:h], y[:h]).predict(raw[h:])
        m = calib_metrics(y[h:], p)
        assert m["brier"] < base["brier"] and m["ece"] < base["ece"], method
        assert p.min() > 0 and p.max() < 1

def test_isotonic_is_monotone_and_none_is_identity():
    raw, y = distorted()
    c = Calibrator("isotonic").fit(raw, y)
    grid = np.linspace(0.01, 0.99, 50)
    assert (np.diff(c.predict(grid)) >= -1e-12).all()
    assert np.allclose(Calibrator("none").predict(grid), grid)

def test_ece_zero_for_perfect_and_positive_for_biased():
    rng = np.random.default_rng(1); p = rng.uniform(0, 1, 20000); y = (rng.random(20000) < p).astype(int)
    assert ece(y, p) < 0.03 and ece(y, np.clip(p + 0.2, 0, 1)) > 0.1
