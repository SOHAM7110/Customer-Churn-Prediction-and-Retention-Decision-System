"""Stage 3 baseline: 'no purchase in 60+ days => churn'. Every model must beat this clearly.
Test split is excluded by default so the test set stays untouched until the final Stage 4 evaluation."""
import argparse
import pathlib
import pandas as pd
from src.metrics import evaluate, rule_metrics

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS = ROOT / "data/interim/features.parquet"
OUT = ROOT / "results/stage3_baseline.csv"
RULE_DAYS = 60


def main(include_test: bool = False):
    df = pd.read_parquet(FEATS)
    splits = ["train", "val"] + (["test"] if include_test else [])
    rows = []
    for s in splits:
        d = df[df.split == s]
        m = evaluate(d.churn, d.recency_days)                        # recency as a continuous score
        m.update(rule_metrics(d.churn, d.recency_days >= RULE_DAYS))  # the hard rule
        rows.append({"split": s, **m})
    res = pd.DataFrame(rows).round(3)
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"Baseline: recency >= {RULE_DAYS}d => churn\n", res.T.to_string(header=False))

    tr = df[df.split == "train"]
    sweep = [{"rule_days": k, **rule_metrics(tr.churn, tr.recency_days >= k)} for k in (30, 45, 60, 75, 90, 120)]
    print("\nThreshold sweep on TRAIN (context only; the rule stays at 60d):")
    print(pd.DataFrame(sweep).round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true", help="only for the final report")
    main(ap.parse_args().include_test)
