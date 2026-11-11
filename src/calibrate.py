"""Stage 5: calibration (fitted on VALIDATION only) + segment checks.

  python -m src.calibrate                # choose + fit calibrator on validation (time-split inside validation)
  python -m src.calibrate --report-test  # frozen method: reliability + Brier + segment checks on TEST (reporting only)
"""
import argparse
import json
import pathlib
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.features import FEATURES

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS = ROOT / "data/interim/features.parquet"
MODELS, RESULTS = ROOT / "models", ROOT / "results"
METHODS = ["none", "sigmoid", "isotonic"]
EPS = 1e-4


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """none | sigmoid (Platt) | isotonic. Fit on held-out validation scores, never on test."""
    def __init__(self, method="isotonic"):
        assert method in METHODS
        self.method = method

    def fit(self, p, y):
        if self.method == "sigmoid":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "none":
            return p
        out = self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1] if self.method == "sigmoid" else self.m.predict(p)
        return np.clip(out, EPS, 1 - EPS)


def _bins(p, bins=10):
    return pd.qcut(pd.Series(np.asarray(p)).rank(method="first"), bins, labels=False)


def ece(y, p, bins=10) -> float:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "b": _bins(p, bins)})
    g = d.groupby("b").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
    return float((g.n / g.n.sum() * (g.p - g.y).abs()).sum())


def calib_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p),
            "mean_pred": p.mean(), "base_rate": y.mean()}


def reliability(y, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "b": _bins(p, bins)})
    return d.groupby("b").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index(drop=True)


def plot_reliability(curves: dict, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", label="perfect")
    for label, (y, p) in curves.items():
        r = reliability(y, p)
        ax.plot(r.pred, r.obs, marker="o", ms=4, label=label)
    ax.set(xlabel="mean predicted P(churn)", ylabel="observed churn rate", title=title)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def load_scorer():
    reg = json.loads((MODELS / "registry.json").read_text())
    model = joblib.load(MODELS / reg["models"][reg["best"]]["file"])
    cpath = MODELS / "calibrator.joblib"
    cal = joblib.load(cpath) if cpath.exists() else Calibrator("none")
    return reg, model, cal


def predict(model, cal, X):
    """-> (raw probability, calibrated probability)"""
    raw = model.predict_proba(X[FEATURES])[:, 1]
    return raw, cal.predict(raw)


def run_fit():
    reg, model, _ = load_scorer()
    df = pd.read_parquet(FEATS)
    va = df[df.split == "val"].reset_index(drop=True)
    raw, y = model.predict_proba(va[FEATURES])[:, 1], va.churn.values
    snaps = sorted(va.snapshot.unique())
    if len(snaps) < 2:
        print("[warn] only one validation snapshot: method selection is in-sample")
        a = b = np.ones(len(va), bool)
    else:
        a = (va.snapshot == snaps[0]).values          # fit on the earlier validation snapshot...
        b = ~a                                        # ...judge on the later one (time-ordered, like production)
    rows, curves = [], {}
    for method in METHODS:
        pb = Calibrator(method).fit(raw[a], y[a]).predict(raw[b])
        rows.append({"method": method, **calib_metrics(y[b], pb)})
        curves[method if method != "none" else "raw (uncalibrated)"] = (y[b], pb)
    sel = pd.DataFrame(rows)
    best = sel.loc[sel.brier.idxmin(), "method"]
    cal = Calibrator(best).fit(raw, y)                # final calibrator uses ALL validation data
    joblib.dump(cal, MODELS / "calibrator.joblib")
    RESULTS.mkdir(exist_ok=True)
    sel.round(4).to_csv(RESULTS / "stage5_method_selection.csv", index=False)
    plot_reliability(curves, RESULTS / "stage5_reliability_val_holdout.png", "Reliability (later validation snapshot)")
    reg["calibration"] = {"method": best, "fit_on": "validation", "selection_brier": sel.set_index("method").brier.round(4).to_dict()}
    (MODELS / "registry.json").write_text(json.dumps(reg, indent=2))
    print(f"Model: {reg['best']}\n", sel.round(4).to_string(index=False))
    print(f"\nChosen: {best} (lowest Brier on the later validation snapshot). Calibrator saved to models/.")
    print("Next: python -m src.explain")


SEG_DIMS = ["country_group", "tenure_bucket", "spend_tier", "order_history"]


def add_segments(df, train):
    q1, q2 = train.monetary.quantile([1 / 3, 2 / 3])          # tier cut-points come from TRAIN
    out = df.copy()
    out["country_group"] = np.where(out.country.isin(["United Kingdom", "EIRE", "Germany", "France"]), out.country, "Other")
    out["tenure_bucket"] = pd.cut(out.tenure_days, [-1, 90, 180, 365, np.inf], labels=["<90d", "90-180d", "180-365d", ">365d"]).astype(str)
    out["spend_tier"] = pd.cut(out.monetary, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"]).astype(str)
    out["order_history"] = pd.cut(out.n_orders_alltime, [0, 1, 4, np.inf], labels=["1 order", "2-4 orders", "5+ orders"]).astype(str)
    return out


def segment_table(d, pcol="p_cal") -> pd.DataFrame:
    rows = []
    for dim in SEG_DIMS:
        for seg, g in d.groupby(dim):
            y, p, n = g.churn.values, g[pcol].values, len(g)
            ok = n >= 50 and y.min() != y.max()
            rows.append({"dimension": dim, "segment": seg, "n": n, "churn_rate": y.mean(), "mean_pred": p.mean(),
                         "calib_gap": p.mean() - y.mean(),
                         "roc_auc": roc_auc_score(y, p) if ok else np.nan,
                         "pr_auc": average_precision_score(y, p) if ok else np.nan,
                         "brier": brier_score_loss(y, np.clip(p, EPS, 1 - EPS)), "note": "small n" if n < 100 else ""})
    return pd.DataFrame(rows)


def run_report_test():
    reg, model, cal = load_scorer()
    if cal.method == "none" and "calibration" not in reg:
        sys.exit("Run `python -m src.calibrate` first.")
    df = pd.read_parquet(FEATS)
    te, tr = df[df.split == "test"].reset_index(drop=True), df[df.split == "train"]
    raw, p = predict(model, cal, te)
    y = te.churn.values
    res = pd.DataFrame([{"version": "raw", **calib_metrics(y, raw)}, {"version": f"calibrated ({cal.method})", **calib_metrics(y, p)}])
    res.round(4).to_csv(RESULTS / "stage5_calibration_test.csv", index=False)
    plot_reliability({"raw": (y, raw), f"calibrated ({cal.method})": (y, p)}, RESULTS / "stage5_reliability_test.png", "Reliability (test)")
    print("== Calibration on TEST (method chosen on validation) ==\n", res.round(4).to_string(index=False))

    seg = segment_table(add_segments(te.assign(p_cal=p), tr))
    seg.round(4).to_csv(RESULTS / "stage5_segments_test.csv", index=False)
    print("\n== Segment checks (test, calibrated) ==\n", seg.round(3).to_string(index=False))
    s = seg.dropna(subset=["roc_auc"]).assign(label=lambda x: x.dimension + ": " + x.segment)
    fig, ax = plt.subplots(1, 2, figsize=(12, max(4, 0.28 * len(s))))
    ax[0].barh(s.label, s.roc_auc); ax[0].axvline(0.5, color="grey", ls=":"); ax[0].set_title("ROC-AUC by segment")
    ax[1].barh(s.label, s.calib_gap); ax[1].axvline(0, color="grey"); ax[1].set_title("mean predicted - observed churn")
    ax[1].set_yticklabels([]); fig.tight_layout(); fig.savefig(RESULTS / "stage5_segments_test.png", dpi=120); plt.close(fig)
    if len(s):
        w = s.loc[s.roc_auc.idxmin()]
        print(f"\nWeakest segment by AUC: {w.label} (AUC {w.roc_auc:.3f}, n={w.n}). Report it in the README.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-test", action="store_true")
    run_report_test() if ap.parse_args().report_test else run_fit()
