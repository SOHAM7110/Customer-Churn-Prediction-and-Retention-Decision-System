import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.decide import unit_profit, evaluate_policy, budget_curve, policy_table, top_k_mask, expected_revenue

E = {"w": 0.25, "m": 0.30, "d": 0.05, "c": 2.0}
def close(a, b, t=1e-9): return abs(a - b) < t

def test_unit_profit_known_value():
    assert close(unit_profit(0.8, 100, E), 0.8 * 0.25 * 0.3 * 100 - (2 + 5))     # = -1.0

def test_contact_threshold_matches_formula():
    R = 1000.0; thr = (E["c"] / R + E["d"]) / (E["w"] * E["m"])                  # 0.69333
    assert unit_profit(thr + 0.01, R, E) > 0 and unit_profit(thr - 0.01, R, E) < 0

def test_evaluate_policy_hand_computed():
    e = {"w": 0.5, "m": 0.4, "d": 0.1, "c": 1.0}
    y, R, mask = np.array([1, 0, 1, 0]), np.array([100., 100., 200., 200.]), np.array([1, 1, 0, 0], bool)
    r = evaluate_policy(mask, y, R, e)
    assert close(r["retained_margin"], 20) and close(r["offer_cost"], 22) and close(r["net_profit"], -2)
    assert close(r["churn_loss_reduction_pct"], 100 * 20 / 120) and r["contacted"] == 2 and close(r["precision"], 0.5)

def test_budget_curve_properties():
    rng = np.random.default_rng(0); n = 500
    p, R = rng.uniform(0, 1, n), rng.gamma(2, 400, n); y = (rng.random(n) < p).astype(int)
    bc = budget_curve(p, R, y, E)
    s = unit_profit(p, R, E)
    assert bc.expected_profit_model.iloc[0] == 0 and (np.diff(bc.contacted) >= 0).all()
    assert close(bc.expected_profit_model.iloc[-1], s[s > 0].sum(), 1e-6)       # full budget = contact every positive customer
    assert (np.diff(bc.expected_profit_model) >= -1e-9).all()                  # only positive-profit customers are added

def test_policy_table_rows_and_nobody_is_zero():
    rng = np.random.default_rng(1); n = 400
    p, R = rng.uniform(0, 1, n), rng.gamma(2, 400, n); y = (rng.random(n) < p).astype(int)
    t = policy_table(p, np.clip(p * 1.3, 0, 1), y, R, E, 0.2).set_index("policy")
    assert t.loc["contact nobody", "net_profit"] == 0 and t.loc["contact everyone", "contacted"] == n
    assert t.loc["optimized, budget 20% of customers", "contacted"] <= int(np.ceil(0.2 * n))
    assert len(t) == 7

def test_top_k_respects_k_and_sign():
    s = np.array([5., -1., 3., 2., -4.]); m = top_k_mask(s, 4)
    assert m.tolist() == [True, False, True, True, False] and top_k_mask(s, 0).sum() == 0

def test_expected_revenue_scaling():
    assert close(expected_revenue(600.0), 300.0)
