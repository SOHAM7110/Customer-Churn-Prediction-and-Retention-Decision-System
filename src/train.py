"""Stage 4: baseline vs logistic regression vs XGBoost vs LightGBM, tracked in MLflow.

  python -m src.train                  # imbalance study -> tuning -> comparison on VAL (test untouched)
  python -m src.train --final-test     # run ONCE at the end: evaluates the saved models on TEST

Validation is used for model choice, tuning, and (Stage 5) calibration. Models are trained on train only.
"""
import argparse
import contextlib
import datetime
import json
import pathlib
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from src import models as M
from src.features import FEATURES
from src.metrics import evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS = ROOT / "data/interim/features.parquet"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
GUARD = RESULTS / ".test_evaluated"
EXPERIMENT = "churn-retention"
SHOW = ["pr_auc", "roc_auc", "lift_top_decile", "recall_at_10pct", "recall_at_20pct", "brier"]


class Tracker:
    """Thin MLflow wrapper: no-ops cleanly if mlflow is missing or disabled."""
    def __init__(self, enabled=True):
        self.mlflow = None
        if enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
                mlflow.set_experiment(EXPERIMENT)
                self.mlflow = mlflow
            except ImportError:
                print("[warn] mlflow not installed: runs will not be tracked")

    @contextlib.contextmanager
    def run(self, name, params=None, nested=False):
        if self.mlflow is None:
            yield self
            return
        with self.mlflow.start_run(run_name=name, nested=nested):
            if params:
                self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            yield self

    def metrics(self, d):
        if self.mlflow:
            self.mlflow.log_metrics({k: float(v) for k, v in d.items() if v is not None and np.isfinite(v)})

    def artifact(self, path):
        if self.mlflow:
            self.mlflow.log_artifact(str(path))


def score(model, d):
    p = model.predict_proba(d[FEATURES])[:, 1]
    m = evaluate(d.churn, p)
    m["brier"] = brier_score_loss(d.churn, p)     # uncalibrated: class weights / resampling distort it (Stage 5 fixes)
    return m, p


def baseline_metrics(d):
    m = evaluate(d.churn, d.recency_days)
    m["brier"] = np.nan                            # recency is a score, not a probability
    return m


def imbalance_study(names, tr, va, tracker):
    rows = []
    for name in names:
        for strat in M.STRATEGIES:
            try:
                model = M.fit_model(name, M.DEFAULTS[name], strat, tr[FEATURES], tr.churn.values)
            except ImportError as e:
                print(f"[skip] {name}: {e}")
                break
            m, _ = score(model, va)
            rows.append({"model": name, "strategy": strat, **m})
            with tracker.run(f"imbalance/{name}/{strat}", {"model": name, "strategy": strat}, nested=True):
                tracker.metrics({f"val_{k}": v for k, v in m.items()})
    return pd.DataFrame(rows)


def tune(name, strat, tr, va, n_trials):
    """Returns (best_params, trials_dataframe_or_None). Objective: validation PR-AUC."""
    Xtr, ytr, Xva, yva = tr[FEATURES], tr.churn.values, va[FEATURES], va.churn.values

    def val_pr(params):
        return average_precision_score(yva, M.fit_model(name, params, strat, Xtr, ytr).predict_proba(Xva)[:, 1])

    if name == "lr":
        best = max(({"C": C} for C in (0.01, 0.1, 1.0, 10.0)), key=val_pr)
        return best, None
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=M.SEED))
    study.optimize(lambda t: val_pr(M.SPACES[name](t)), n_trials=n_trials)
    return {**M.DEFAULTS[name], **study.best_params}, study.trials_dataframe()


def plot_curves(va, preds, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    y = va.churn.values
    series = {"baseline (recency)": va.recency_days.values, **preds}
    for label, s in series.items():
        p, r, _ = precision_recall_curve(y, s)
        ax[0].plot(r, p, label=label)
        order = np.argsort(-np.asarray(s), kind="stable")
        gain = np.cumsum(y[order]) / y.sum()
        ax[1].plot(np.arange(1, len(y) + 1) / len(y), gain, label=label)
    ax[0].axhline(y.mean(), color="grey", ls=":"); ax[0].set(title="Precision-recall (validation)", xlabel="recall", ylabel="precision")
    ax[1].plot([0, 1], [0, 1], color="grey", ls=":"); ax[1].set(title="Cumulative gain: churners reached vs budget",
                                                              xlabel="share of customers contacted", ylabel="share of churners reached")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def run_training(args):
    tracker = Tracker(not args.no_mlflow)
    df = pd.read_parquet(FEATS)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    names = [n.strip() for n in args.models.split(",")]
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

    with tracker.run("stage4_training", {"seed": M.SEED, "n_trials": args.trials, "models": args.models,
                                          "n_train": len(tr), "n_val": len(va)}):
        # A) imbalance handling: class weights vs resampling, default hyper-parameters
        study = imbalance_study(names, tr, va, tracker)
        study.round(4).to_csv(RESULTS / "stage4_imbalance_study.csv", index=False)
        print("\n== Imbalance handling (validation, default params) ==")
        print(study[["model", "strategy"] + SHOW].round(3).to_string(index=False))
        best_strat = study.loc[study.groupby("model").pr_auc.idxmax()].set_index("model").strategy.to_dict()
        print("\nBest strategy per model by val PR-AUC:", best_strat)

        # B) light tuning, C) final fit, save, score
        registry, rows, preds = {}, [{"model": "baseline (recency)", **baseline_metrics(va)}], {}
        for name in best_strat:
            strat = best_strat[name]
            with tracker.run(f"tune/{name}", {"model": name, "strategy": strat}, nested=True):
                params, trials = tune(name, strat, tr, va, args.trials)
                if trials is not None:
                    trials.to_csv(RESULTS / f"stage4_optuna_{name}.csv", index=False)
                    tracker.artifact(RESULTS / f"stage4_optuna_{name}.csv")
            model = M.fit_model(name, params, strat, tr[FEATURES], tr.churn.values)
            m, preds[name] = score(model, va)
            path = MODELS / f"{name}.joblib"
            joblib.dump(model, path)
            registry[name] = {"file": path.name, "strategy": strat, "params": _clean(params), "val": _clean(m)}
            rows.append({"model": name, **m})
            with tracker.run(f"final/{name}", {"model": name, "strategy": strat, **params}, nested=True):
                tracker.metrics({f"val_{k}": v for k, v in m.items()})
                tracker.artifact(path)

        comp = pd.DataFrame(rows)
        base_pr = comp.loc[0, "pr_auc"]
        comp["pr_auc_gain_vs_baseline"] = comp.pr_auc - base_pr
        comp.round(4).to_csv(RESULTS / "stage4_model_comparison.csv", index=False)
        print("\n== Model comparison (validation; tuned on validation, so slightly optimistic) ==")
        print(comp[["model"] + SHOW + ["pr_auc_gain_vs_baseline"]].round(3).to_string(index=False))

        best = comp.iloc[1:].sort_values("pr_auc", ascending=False).iloc[0].model
        plot_curves(va, {n: p for n, p in preds.items()}, RESULTS / "stage4_curves.png")
        tracker.artifact(RESULTS / "stage4_curves.png"); tracker.artifact(RESULTS / "stage4_model_comparison.csv")
        (MODELS / "registry.json").write_text(json.dumps({"features": FEATURES, "best": best, "models": registry}, indent=2))
        print(f"\nBest on validation: {best}. Saved models/, results/stage4_*.csv and stage4_curves.png")
        print("Test set untouched. Next: python -m src.calibrate, then python -m src.explain, then `make final`.")


def run_final_test(force=False):
    if GUARD.exists() and not force:
        sys.exit(f"Test set was already evaluated ({GUARD.read_text().strip()}). Re-running invites test-set tuning. "
                 "Use --force only for a documented reason.")
    reg = json.loads((MODELS / "registry.json").read_text())
    te = pd.read_parquet(FEATS).query("split == 'test'")
    rows = [{"model": "baseline (recency)", **baseline_metrics(te)}]
    for name, info in reg["models"].items():
        m, _ = score(joblib.load(MODELS / info["file"]), te)
        rows.append({"model": name, **m})
    res = pd.DataFrame(rows)
    res["pr_auc_gain_vs_baseline"] = res.pr_auc - res.loc[0, "pr_auc"]
    res.round(4).to_csv(RESULTS / "stage4_test_results.csv", index=False)
    GUARD.write_text(f"{datetime.datetime.now().isoformat(timespec='seconds')} best_on_val={reg['best']}")
    print(f"== TEST (selected on validation: {reg['best']}; churn rate {te.churn.mean():.3f}, n={len(te)}) ==")
    print(res[["model"] + SHOW + ["pr_auc_gain_vs_baseline"]].round(3).to_string(index=False))
    print("\nReport these as-is. Do not retune based on them.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lr,xgb,lgbm")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_final_test(a.force) if a.final_test else run_training(a)
