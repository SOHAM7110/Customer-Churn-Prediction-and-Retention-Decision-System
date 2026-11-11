"""Optional scoring endpoint.   uvicorn app.api:app --port 8000
POST /predict  {"features": {"recency_days": 80, "frequency": 3, ...}}   (missing features are treated as NaN)"""
import pathlib
import sys
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibrate import load_scorer, predict
from src.decide import econ, expected_revenue, unit_profit
from src.features import FEATURES

app = FastAPI(title="Churn retention scoring")
REG, MODEL, CAL = load_scorer()
E = econ()


class PredictRequest(BaseModel):
    features: Dict[str, Optional[float]]


@app.get("/health")
def health():
    return {"status": "ok", "model": REG["best"], "calibration": CAL.method}


@app.post("/predict")
def predict_one(req: PredictRequest):
    row = pd.DataFrame([{f: req.features.get(f) for f in FEATURES}]).astype(float)
    raw, cal = predict(MODEL, CAL, row)
    R = float(expected_revenue(row.monetary.fillna(0.0)).iloc[0])
    profit = float(unit_profit(cal[0], R, E))
    return {"p_churn_raw": float(raw[0]), "p_churn_calibrated": float(cal[0]), "expected_revenue_90d": R,
            "expected_profit_if_contacted": profit, "recommended_action": "contact" if profit > 0 else "do_not_contact",
            "note": "offer success rate is an assumption (config.yaml)"}
