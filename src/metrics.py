"""Shared evaluation metrics (used by the baseline now, and by every model in Stage 4+).
Accuracy is deliberately absent: with a minority positive class it is misleading."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(y, score, budgets=(0.10, 0.20)) -> dict:
    """Ranking metrics. `score` = anything where higher means more likely to churn."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    assert y.min() != y.max(), "need both classes to evaluate"
    n, base = len(y), y.mean()
    order = np.argsort(-score, kind="stable")
    out = {"n": n, "churn_rate": base,
           "roc_auc": roc_auc_score(y, score),
           "pr_auc": average_precision_score(y, score)}
    for b in budgets:
        k = int(np.ceil(b * n))
        top = y[order[:k]]
        out[f"recall_at_{int(round(b * 100))}pct"] = top.sum() / y.sum()   # share of all churners reached
        if abs(b - 0.10) < 1e-9:
            out["lift_top_decile"] = top.mean() / base                    # precision in top 10% / base rate
    return out


def rule_metrics(y, pred) -> dict:
    """Hard 0/1 predictions (e.g. the baseline rule)."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    flagged = int(pred.sum())
    return {"flagged_rate": flagged / len(y),
            "precision": tp / flagged if flagged else float("nan"),
            "recall": tp / int(y.sum())}
