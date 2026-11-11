"""Stage 7: A/B test SIMULATION for the optimized policy.  *** SIMULATION: the effect size is ASSUMED, not measured. ***

Pool   = test customers the optimized policy would contact.
Design = randomise the pool 50/50 into treatment (offer) / control (nothing).
Metric = realised 90-day margin per customer:
   control   : (1 - y) * m * R
   treatment : (1 - y + y * Z) * m * R - (c + d * R),   Z ~ Bernoulli(w)   (a would-be churner is retained with prob w)
What it proves: the machinery (power, CI) works and how big a real test must be. What it can NOT prove: that w is real.
"""
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.decide import econ, expected_revenue, top_k_mask, unit_profit
from src.labels import load_config

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATS, RESULTS = ROOT / "data/interim/features.parquet", ROOT / "results"


def required_n_per_arm(delta, sigma, alpha=0.05, power=0.80):
    if delta <= 0:
        return float("inf")
    return 2 * ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sigma / delta) ** 2


def power_at(n, delta, sigma, alpha=0.05):
    return float(stats.norm.cdf(abs(delta) / (sigma * np.sqrt(2 / np.asarray(n, float))) - stats.norm.ppf(1 - alpha / 2)))


def outcomes(y, R, e, treated, rng):
    z = rng.random(len(y)) < e["w"]
    ctrl = (1 - y) * e["m"] * R
    treat = (1 - y + y * z) * e["m"] * R - (e["c"] + e["d"] * R)
    return np.where(treated, treat, ctrl)


def one_trial(y, R, e, rng):
    n = len(y)
    perm = rng.permutation(n)
    half = n // 2
    t, c = perm[:half], perm[half:2 * half]
    yt = outcomes(y[t], R[t], e, np.ones(half, bool), rng)
    yc = outcomes(y[c], R[c], e, np.zeros(half, bool), rng)
    return yt, yc


def main():
    from src.calibrate import load_scorer, predict
    cfg, ab = load_config(), load_config()["ab"]
    reg, model, cal = load_scorer()
    te = pd.read_parquet(FEATS).query("split == 'test'").reset_index(drop=True)
    _, p = predict(model, cal, te)
    e, R, y_all = econ(cfg), expected_revenue(te.monetary), te.churn.values.astype(float)
    mask, note = unit_profit(p, R, e) > 0, "optimized policy"
    if mask.sum() < 200:
        mask, note = top_k_mask(unit_profit(p, R, e), 0.2 * len(p), positive_only=False), "top 20% by expected profit (optimized policy contacts too few)"
    y, Rp = y_all[mask], R[mask]
    rng = np.random.default_rng(ab["seed"])
    print(f"*** SIMULATION (assumed P(offer works) = {e['w']}) ***\nPool: {mask.sum():,} customers ({note})")

    yt, yc = one_trial(y, Rp, e, rng)
    diff = yt.mean() - yc.mean()
    boot = np.array([rng.choice(yt, len(yt)).mean() - rng.choice(yc, len(yc)).mean() for _ in range(ab["n_boot"])])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sigma = float(np.sqrt((yt.var(ddof=1) + yc.var(ddof=1)) / 2))
    delta = float(np.mean(e["w"] * y * e["m"] * Rp - e["c"] - e["d"] * Rp))          # expected true effect per treated customer
    n_arm = len(yt)
    hits = 0
    for _ in range(ab["n_sims"]):
        a, b = one_trial(y, Rp, e, rng)
        r = stats.ttest_ind(a, b, equal_var=False)                       # two-sided, same convention as power_at()
        hits += bool(r.pvalue < ab["alpha"] and r.statistic > 0)
    need = required_n_per_arm(delta, sigma, ab["alpha"], ab["power"])

    summ = {"pool_size": int(mask.sum()), "n_per_arm": n_arm, "assumed_p_offer_works": e["w"],
            "expected_effect_per_customer_gbp": delta, "measured_diff_per_customer_gbp": diff,
            "boot_ci95_low": lo, "boot_ci95_high": hi,
            "total_incremental_profit_gbp": diff * n_arm, "total_ci95_low": lo * n_arm, "total_ci95_high": hi * n_arm,
            "sd_per_customer_gbp": sigma, "empirical_power": hits / ab["n_sims"], "analytic_power": power_at(n_arm, delta, sigma, ab["alpha"]),
            "n_per_arm_for_target_power": need, "target_power": ab["power"]}
    RESULTS.mkdir(exist_ok=True)
    pd.Series(summ).to_csv(RESULTS / "stage7_ab_simulation.csv", header=["value"])
    for k, v in summ.items():
        print(f"  {k:36s} {v:,.4g}" if isinstance(v, float) else f"  {k:36s} {v}")
    verdict = ("UNDERPOWERED at this pool size" if summ["empirical_power"] < ab["power"] else "adequately powered at this pool size")
    print(f"\nVerdict: {verdict}. A real test needs about {need:,.0f} customers per arm for {ab['power']:.0%} power "
          f"(this assumes the effect is real; if w is lower the requirement grows quickly).")

    grid = np.logspace(2, 6, 60)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(grid, [power_at(n, delta, sigma, ab["alpha"]) for n in grid])
    ax.axhline(ab["power"], color="grey", ls=":"); ax.axvline(n_arm, color="tab:red", ls="--", label=f"simulated arm size ({n_arm:,})")
    ax.set(xlabel="customers per arm", ylabel="power", title="A/B simulation: power vs sample size (ASSUMED effect)"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "stage7_ab_power.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
