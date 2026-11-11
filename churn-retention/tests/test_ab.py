import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.simulate_ab import required_n_per_arm, power_at, one_trial

def test_required_n_known_value_and_power_roundtrip():
    n = required_n_per_arm(1.0, 1.0, 0.05, 0.80)
    assert abs(n - 15.7) < 0.1
    assert abs(power_at(n, 1.0, 1.0, 0.05) - 0.80) < 1e-6

def test_no_effect_means_infinite_sample():
    assert required_n_per_arm(0.0, 1.0) == float("inf") and required_n_per_arm(-1.0, 1.0) == float("inf")

def test_power_increases_with_n_and_effect():
    assert power_at(100, 1, 5) < power_at(1000, 1, 5) < power_at(10000, 1, 5)
    assert power_at(500, 1, 5) < power_at(500, 2, 5)

def test_simulated_effect_matches_expectation():
    rng = np.random.default_rng(0); n = 40000
    y = (rng.random(n) < 0.6).astype(float); R = rng.gamma(2, 300, n)
    e = {"w": 0.3, "m": 0.3, "d": 0.05, "c": 2.0}
    diffs = []
    for _ in range(20):
        yt, yc = one_trial(y, R, e, rng); diffs.append(yt.mean() - yc.mean())
    expected = np.mean(e["w"] * y * e["m"] * R - e["c"] - e["d"] * R)
    assert abs(np.mean(diffs) - expected) < 0.35 * abs(expected) + 1.0
