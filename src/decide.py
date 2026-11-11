"""Stage 7 decision layer: expected profit per customer, contact policies, sensitivity, budget curve.

Per customer i (R = expected 90-day revenue, p = calibrated P(churn)):
    expected profit of contacting = p * w * m * R  -  (c + d * R)
      w = P(offer works), m = gross margin, c = fixed contact cost, d = discount paid on R by every contacted customer
    => contact iff p > (c/R + d) / (w * m)        (a per-customer threshold, not a global 0.5)

Policies are scored against ACTUAL test-set churn labels with the ASSUMED offer effect w:
    realised profit = sum over contacted of ( y * w * m * R - c - d * R )
"""
import json
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.labels import load_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS, RESULTS = ROOT / "data/interim/features.parquet", ROOT / "results"


def econ(cfg=None, **over) -> dict:
    e = (cfg or load_config())["economics"]
    out = {"w": e["p_offer_works"], "m": e["gross_margin"], "d": e["discount_pct"], "c": e["contact_cost_gbp"]}
    out.update(over)
    return out


def expected_revenue(monetary, obs_days=180, horizon=90):
    return np.asarray(monetary, float) * horizon / obs_days


def unit_profit(p, R, e):
    p, R = np.asarray(p, float), np.asarray(R, float)
    return p * e["w"] * e["m"] * R - (e["c"] + e["d"] * R)


def realized_unit_profit(y, R, e):
    return unit_profit(np.asarray(y, float), R, e)          # same formula with the true label in place of p


def top_k_mask(score, k, positive_only=True):
    score, mask, k = np.asarray(score, float), np.zeros(len(score), bool), int(k)
    if k > 0:
        mask[np.argsort(-score, kind="stable")[:k]] = True
        if positive_only:
            mask &= score > 0
    return mask


def evaluate_policy(mask, y, R, e) -> dict:
    mask, y, R = np.asarray(mask, bool), np.asarray(y, float), np.asarray(R, float)
    n, k = len(y), int(mask.sum())
    loss_total = float((y * e["m"] * R).sum())                    # margin lost to churners if we do nothing
    retained = float((mask * y * e["w"] * e["m"] * R).sum())
    cost = float((mask * (e["c"] + e["d"] * R)).sum())
    return {"contacted": k, "contact_rate": k / n, "precision": float(y[mask].mean()) if k else np.nan,
            "retained_margin": retained, "offer_cost": cost, "net_profit": retained - cost,
            "churn_loss_reduction_pct": 100 * retained / loss_total if loss_total else np.nan,
            "net_loss_reduction_pct": 100 * (retained - cost) / loss_total if loss_total else np.nan}


def random_same_size(y, R, e, k) -> dict:
    """Expected result of contacting k customers chosen at random."""
    base = evaluate_policy(np.ones(len(y), bool), y, R, e)
    f = k / len(y)
    out = {key: (v * f if key in ("retained_margin", "offer_cost", "net_profit") else v) for key, v in base.items()}
    out.update(contacted=k, contact_rate=f, precision=float(np.mean(y)))
    for key in ("churn_loss_reduction_pct", "net_loss_reduction_pct"):
        out[key] = base[key] * f
    return out


def policy_table(p_cal, p_raw, y, R, e, budget_frac) -> pd.DataFrame:
    n = len(y)
    s = unit_profit(p_cal, R, e)
    opt = s > 0
    masks = {"contact nobody": np.zeros(n, bool), "contact everyone": np.ones(n, bool),
             "default threshold (p >= 0.5)": np.asarray(p_cal) >= 0.5,
             "optimized: expected profit > 0": opt,
             f"optimized, budget {budget_frac:.0%} of customers": top_k_mask(s, np.ceil(budget_frac * n)),
             "optimized on UNcalibrated probs": unit_profit(p_raw, R, e) > 0}
    rows = [{"policy": k, **evaluate_policy(m, y, R, e)} for k, m in masks.items()]
    rows.append({"policy": "random, same size as optimized", **random_same_size(y, R, e, int(opt.sum()))})
    return pd.DataFrame(rows)


def sensitivity(p, y, R, cfg) -> pd.DataFrame:
    base = econ(cfg)
    fixed, thr, allc = unit_profit(p, R, base) > 0, np.asarray(p) >= 0.5, np.ones(len(y), bool)
    rows = []
    for m in cfg["sensitivity"]["gross_margin"]:
        for w in cfg["sensitivity"]["p_offer_works"]:
            for d in cfg["sensitivity"]["discount_pct"]:
                e = econ(cfg, w=w, d=d, m=m)
                r = realized_unit_profit(y, R, e)
                rows.append({"margin": m, "p_offer_works": w, "discount_pct": d, "contact_everyone": r.sum(),
                             "threshold_0.5": r[thr].sum(), "optimized_base_assumptions": r[fixed].sum(),
                             "optimized_reoptimized": r[unit_profit(p, R, e) > 0].sum()})
    return pd.DataFrame(rows)


def plot_sensitivity(sens, path):
    ms = sorted(sens.margin.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 3.8))
    axes = np.atleast_1d(axes)
    sens = sens.assign(gain=sens.optimized_base_assumptions - sens["threshold_0.5"])
    lim = max(1.0, float(sens.gain.abs().max()))
    for ax, m in zip(axes, ms):
        g = sens[sens.margin == m].pivot(index="discount_pct", columns="p_offer_works", values="gain")
        im = ax.imshow(g.values, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(g.columns))); ax.set_xticklabels(g.columns); ax.set_yticks(range(len(g.index))); ax.set_yticklabels(g.index)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:,.0f}", ha="center", va="center", fontsize=7)
        ax.set(title=f"margin={m:.0%}", xlabel="P(offer works)", ylabel="discount")
    fig.suptitle("Net profit GBP: optimized policy (base assumptions) minus default 0.5 threshold", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def budget_curve(p, R, y, e, n_points=101) -> pd.DataFrame:
    """Top-K by expected profit, K swept from 0 to 100% of customers (only positive-profit customers are ever contacted)."""
    s, r = unit_profit(p, R, e), realized_unit_profit(y, R, e)
    o = np.argsort(-s, kind="stable")
    pos = s[o] > 0
    cs_model, cs_back = np.cumsum(np.where(pos, s[o], 0.0)), np.cumsum(np.where(pos, r[o], 0.0))
    fr = np.linspace(0, 1, n_points)
    k = np.ceil(fr * len(s)).astype(int)
    pad = lambda cs: np.concatenate([[0.0], cs])[k]
    return pd.DataFrame({"budget_frac": fr, "contacted": np.minimum(k, int(pos.sum())),
                         "expected_profit_model": pad(cs_model), "backtest_profit": pad(cs_back)})


def main():
    from src.calibrate import load_scorer, predict
    cfg = load_config()
    reg, model, cal = load_scorer()
    te = pd.read_parquet(FEATS).query("split == 'test'").reset_index(drop=True)
    raw, p = predict(model, cal, te)
    y, R, e = te.churn.values, expected_revenue(te.monetary), econ(cfg)
    RESULTS.mkdir(exist_ok=True)
    print(f"Test customers-snapshots: {len(te):,}; churn rate {y.mean():.3f}; calibration: {cal.method}")
    print(f"Assumptions (NOT measured): P(offer works)={e['w']}, margin={e['m']}, discount={e['d']}, contact cost={e['c']}")

    tab = policy_table(p, raw, y, R, e, cfg["budget"]["contact_fraction"])
    tab.round(4).to_csv(RESULTS / "stage7_policy_comparison.csv", index=False)
    show = ["policy", "contacted", "contact_rate", "precision", "net_profit", "net_loss_reduction_pct"]
    print("\n== Policy comparison (backtest on test, assumed offer effect) ==\n", tab[show].round(3).to_string(index=False))
    d = tab.set_index("policy")
    opt, thr = d.loc["optimized: expected profit > 0"], d.loc["default threshold (p >= 0.5)"]
    print(f"\nOptimized vs default 0.5: net profit GBP {opt.net_profit:,.0f} vs {thr.net_profit:,.0f} "
          f"(gain {opt.net_profit - thr.net_profit:,.0f}); churn loss reduced by {opt.net_loss_reduction_pct:.1f}% (net) vs {thr.net_loss_reduction_pct:.1f}%")

    sens = sensitivity(p, y, R, cfg)
    sens.round(2).to_csv(RESULTS / "stage7_sensitivity.csv", index=False)
    plot_sensitivity(sens, RESULTS / "stage7_sensitivity.png")
    fx, th, ev = sens.optimized_base_assumptions, sens["threshold_0.5"], sens.contact_everyone
    print(f"\n== Sensitivity: {len(sens)} assumption cells ==")
    print(f"optimized beats 0.5-threshold in {(fx > th).mean():.0%}, beats contact-everyone in {(fx > ev).mean():.0%}, "
          f"is profitable (>0) in {(fx > 0).mean():.0%} of cells")

    bc = budget_curve(p, R, y, e)
    bc.round(2).to_csv(RESULTS / "stage7_budget_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bc.budget_frac * 100, bc.expected_profit_model, label="expected (model probabilities)")
    ax.plot(bc.budget_frac * 100, bc.backtest_profit, label="backtest (actual churn, assumed effect)")
    ax.set(xlabel="% of customers contacted (top by expected profit)", ylabel="net profit GBP", title="Profit vs contact budget"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "stage7_budget_curve.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
