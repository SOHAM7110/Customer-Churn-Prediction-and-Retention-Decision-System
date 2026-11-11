"""Stage 8: bundle everything the demo app needs (scored test customers + precomputed explanations),
so the deployed app needs no raw data, no SHAP library and no retraining.   python -m src.export_app_data"""
import json
import pathlib
import sys

import pandas as pd

from src.calibrate import load_scorer, predict
from src.decide import expected_revenue
from src.explain import contributions
from src.features import FEATURES

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "app/data"


def main():
    reg, model, cal = load_scorer()
    te = pd.read_parquet(ROOT / "data/interim/features.parquet").query("split == 'test'").reset_index(drop=True)
    raw, p = predict(model, cal, te)
    sv, base = contributions(model, te)
    out = te[["customer_id", "snapshot", "country", "churn"] + FEATURES].copy()
    out["p_raw"], out["p_cal"], out["R90"] = raw, p, expected_revenue(te.monetary)
    for f in FEATURES:
        out[f"shap__{f}"] = sv[f].values
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "customers.parquet")
    (OUT / "meta.json").write_text(json.dumps({
        "model": reg["best"], "calibration": cal.method, "base_logodds": base,
        "snapshots": [str(s.date()) for s in sorted(te.snapshot.unique())], "n_rows": len(out)}, indent=2))
    print(f"Wrote {OUT / 'customers.parquet'} ({len(out):,} rows, {out.shape[1]} cols, model={reg['best']}, calibration={cal.method})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
