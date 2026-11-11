"""Stage 6: SHAP-style explanations. Works for the winning model of either family:
 - trees (XGBoost / LightGBM): shap.TreeExplainer (imported lazily so the app never needs shap)
 - logistic regression: exact linear SHAP, coef * (x_scaled - mean_scaled), no library needed
Contributions are on the model's LOG-ODDS scale (before calibration)."""
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import FEATURES
from src.models import LOG_COLS

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS, RESULTS = ROOT / "data/interim/features.parquet", ROOT / "results"

FRIENDLY = {
    "recency_days": "Days since last order", "frequency": "Orders (180d)", "monetary": "Spend (180d)",
    "aov": "Avg order value", "orders_30": "Orders (30d)", "orders_60": "Orders (60d)", "orders_90": "Orders (90d)",
    "revenue_30": "Spend (30d)", "revenue_60": "Spend (60d)", "revenue_90": "Spend (90d)",
    "order_trend": "Order trend (recent vs earlier)", "spend_trend": "Spend trend (recent vs earlier)",
    "mean_gap_days": "Avg days between orders", "std_gap_days": "Variability of gaps", "gap_cv": "Gap variability (relative)",
    "recency_over_gap": "Overdue vs own rhythm", "avg_items_per_order": "Items per order",
    "n_unique_products": "Distinct products", "return_rate": "Return rate", "tenure_days": "Tenure (days)",
    "n_orders_alltime": "Lifetime orders", "is_uk": "UK customer"}


def lr_contributions(pipe, X):
    ct, imp, sc, clf = (pipe.named_steps[k] for k in ("log", "impute", "scale", "clf"))
    Z = sc.transform(imp.transform(ct.transform(X[FEATURES])))
    base_names = LOG_COLS + [c for c in FEATURES if c not in LOG_COLS]       # ColumnTransformer output order
    miss = list(imp.indicator_.features_) if imp.indicator_ is not None else []
    coef = clf.coef_[0]
    mu = Z.mean(axis=0)
    phi = coef * (Z - mu)
    base = float(clf.intercept_[0] + (coef * mu).sum())
    nb = len(base_names)
    out = pd.DataFrame(phi[:, :nb], columns=base_names, index=X.index)
    for j, i in enumerate(miss):                                              # fold "_missing" flags into their feature
        out[base_names[i]] += phi[:, nb + j]
    return out[FEATURES], base


def tree_contributions(model, X):
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X[FEATURES])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return pd.DataFrame(sv, columns=FEATURES, index=X.index), float(np.ravel(ex.expected_value)[-1])


def contributions(model, X):
    """-> (DataFrame of per-feature log-odds contributions, base value)"""
    return lr_contributions(model, X) if hasattr(model, "named_steps") else tree_contributions(model, X)


def top_reasons(shap_row, x_row, k=5):
    top = shap_row.reindex(FEATURES).abs().sort_values(ascending=False).index[:k]
    out = []
    for f in top:
        s, v = float(shap_row[f]), x_row[f]
        val = "n/a" if pd.isna(v) else f"{v:,.3g}"
        out.append({"feature": f, "label": FRIENDLY.get(f, f), "value": v, "shap": s,
                    "text": f"{FRIENDLY.get(f, f)} = {val}: {'raises' if s > 0 else 'lowers'} churn risk"})
    return out


def plot_global(sv, X, path, top=12):
    order = sv.abs().mean().sort_values(ascending=False).index[:top][::-1]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].barh([FRIENDLY[f] for f in order], [sv[f].abs().mean() for f in order])
    ax[0].set_title("Mean |contribution| (log-odds)")
    for i, f in enumerate(order):
        c = X[f].rank(pct=True).fillna(0.5)
        ax[1].scatter(sv[f], i + rng.uniform(-0.3, 0.3, len(sv)), c=c, cmap="coolwarm", s=4, alpha=0.5)
    ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels([FRIENDLY[f] for f in order])
    ax[1].axvline(0, color="grey"); ax[1].set_title("Per-customer contributions (red = high feature value)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_local(rows, path):
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (title, reasons) in zip(axes, rows):
        r = reasons[::-1]
        ax.barh([x["label"] for x in r], [x["shap"] for x in r], color=["tab:red" if x["shap"] > 0 else "tab:blue" for x in r])
        ax.axvline(0, color="grey"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    from src.calibrate import load_scorer, predict
    reg, model, cal = load_scorer()
    df = pd.read_parquet(FEATS)
    va = df[df.split == "val"]
    samp = va.sample(min(3000, len(va)), random_state=0)
    sv, base = contributions(model, samp)
    RESULTS.mkdir(exist_ok=True)
    imp = sv.abs().mean().sort_values(ascending=False).rename("mean_abs_contribution")
    imp.round(4).to_csv(RESULTS / "stage6_mean_abs_shap.csv")
    plot_global(sv, samp[FEATURES], RESULTS / "stage6_shap_global.png")
    print(f"Explaining: {reg['best']} (validation sample n={len(samp)})\n", imp.round(3).head(10).to_string())

    corr = samp[FEATURES].corr(method="spearman").abs()
    pairs = [(a, b, round(corr.loc[a, b], 2)) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:] if corr.loc[a, b] > 0.8]
    pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_spearman"]).to_csv(RESULTS / "stage6_correlated_features.csv", index=False)
    print(f"\n{len(pairs)} feature pairs with |Spearman| > 0.8 -> importance is shared between them (README caveat).")

    te = df[df.split == "test"].reset_index(drop=True)
    if len(te):
        raw, p = predict(model, cal, te)
        sv_t, _ = contributions(model, te)
        picks = [("Highest risk", int(np.argmax(p))), ("Median risk", int(np.argsort(p)[len(p) // 2])), ("Lowest risk", int(np.argmin(p)))]
        plot_local([(f"{n}: P(churn)={p[i]:.0%}", top_reasons(sv_t.iloc[i], te.loc[i], 6)) for n, i in picks],
                   RESULTS / "stage6_shap_local_examples.png")
        print("Saved local explanation examples (test customers).")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
