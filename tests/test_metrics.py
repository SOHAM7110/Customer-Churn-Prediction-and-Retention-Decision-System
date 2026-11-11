import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from src.metrics import evaluate, rule_metrics

def test_perfect_ranking():
    y = np.array([1] * 20 + [0] * 80); score = np.arange(100, 0, -1)
    m = evaluate(y, score)
    assert abs(m["roc_auc"] - 1) < 1e-9 and abs(m["pr_auc"] - 1) < 1e-9
    assert abs(m["recall_at_10pct"] - 0.5) < 1e-9      # top 10 of 20 churners
    assert abs(m["lift_top_decile"] - 5.0) < 1e-9      # 100% precision / 20% base rate
    assert abs(m["recall_at_20pct"] - 1.0) < 1e-9

def test_rule_metrics():
    r = rule_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert r["flagged_rate"] == 0.5 and r["precision"] == 0.5 and r["recall"] == 0.5
