# %% [markdown]
# # Stage 1 - EDA + framing evidence
# Goal: produce the numbers that justify the label windows in docs/problem_statement.md.
# Run cell by cell (VS Code / Jupytext) or from the repo root: `python notebooks/01_eda.py`.

# %%
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path("..").resolve()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import load_raw, build_orders

OUT = ROOT / "results/eda"; OUT.mkdir(parents=True, exist_ok=True)
df = load_raw()
print(df.shape); print(df.dtypes)

# %% [markdown]
# ## 1. Data quality: what are we throwing away, and is it safe?
# %%
dq = {
    "rows": len(df),
    "date_min": df.invoice_date.min(), "date_max": df.invoice_date.max(),
    "missing_customer_id_%": round(df.customer_id.isna().mean() * 100, 1),
    "cancellation_lines_%": round(df.is_cancel.mean() * 100, 1),
    "qty<=0_lines_%": round((df.quantity <= 0).mean() * 100, 1),
    "price<=0_lines_%": round((df.price <= 0).mean() * 100, 1),
    "countries": df.country.nunique(),
}
pd.Series(dq).to_csv(OUT / "data_quality.csv"); print(pd.Series(dq))

# %% [markdown]
# ## 2. Customer-level table (genuine purchases only)
# %%
orders = build_orders(df)
n_cust = orders.customer_id.nunique()
print(f"{len(orders):,} orders, {n_cust:,} customers")
per_cust = orders.groupby("customer_id").agg(
    n_orders=("invoice", "count"), total_rev=("revenue", "sum"),
    first=("date", "min"), last=("date", "max"), country=("country", "first"))
print(per_cust.n_orders.describe())
print("share with 1 order: %.1f%%" % ((per_cust.n_orders == 1).mean() * 100))
print("UK share of customers: %.1f%%" % ((per_cust.country == "United Kingdom").mean() * 100))
top1 = per_cust.total_rev.nlargest(max(1, n_cust // 100)).sum() / per_cust.total_rev.sum()
print("revenue share of top 1%% customers: %.1f%%" % (top1 * 100))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
per_cust.n_orders.clip(upper=40).hist(bins=40, ax=ax[0]); ax[0].set_title("Orders per customer (clipped at 40)")
np.log10(per_cust.total_rev.clip(lower=1)).hist(bins=40, ax=ax[1]); ax[1].set_title("log10 lifetime revenue")
fig.tight_layout(); fig.savefig(OUT / "customer_distributions.png", dpi=120)

# %% [markdown]
# ## 3. Purchase rhythm: how long is "a long gap"?
# The inter-order gap distribution is the evidence for the prediction window.
# %%
o = orders.copy()
o["gap_days"] = o.groupby("customer_id").date.diff().dt.total_seconds() / 86400
gaps = o.gap_days.dropna()
gap_pcts = gaps.quantile([.25, .5, .75, .9, .95, .99]).round(1)
print(gap_pcts); gap_pcts.to_csv(OUT / "gap_percentiles.csv")

fig, ax = plt.subplots(figsize=(6, 3.5))
gaps.clip(upper=365).hist(bins=73, ax=ax); ax.set_title("Days between consecutive orders (clipped at 365)")
for d in (60, 90): ax.axvline(d, color="r", ls="--")
fig.tight_layout(); fig.savefig(OUT / "gap_hist.png", dpi=120)

# %% [markdown]
# ## 4. Churn rate under candidate windows
# Active at T = >=1 order in the previous 180 days. Churn = no order in [T, T+W).
# Pick W so churn is a meaningful class share and the window is business-sensible.
# %%
OBS = 180
end = orders.date.max()
snaps = pd.date_range("2010-06-01", "2011-09-01", freq="MS")

def churn_rate(T, W):
    T = pd.Timestamp(T)
    hist = orders[(orders.date < T) & (orders.date >= T - pd.Timedelta(days=OBS))]
    active = set(hist.customer_id)
    fut = orders[(orders.date >= T) & (orders.date < T + pd.Timedelta(days=W))]
    returned = active & set(fut.customer_id)
    return len(active), 1 - len(returned) / len(active)

rows = []
for W in (30, 60, 90, 120):
    for T in snaps:
        if T + pd.Timedelta(days=W) <= end + pd.Timedelta(days=1):   # label window must be fully observed
            n, r = churn_rate(T, W)
            rows.append({"W": W, "snapshot": T.date(), "n_active": n, "churn_rate": round(r, 3)})
cr = pd.DataFrame(rows); cr.to_csv(OUT / "churn_rate_by_window.csv", index=False)
print(cr.pivot(index="snapshot", columns="W", values="churn_rate"))

fig, ax = plt.subplots(figsize=(7, 3.5))
for W, g in cr.groupby("W"): ax.plot(pd.to_datetime(g.snapshot), g.churn_rate, marker="o", label=f"{W}d")
ax.set_title("Churn rate by snapshot and prediction window"); ax.legend(); fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(OUT / "churn_rate_by_window.png", dpi=120)

# %% [markdown]
# ## 5. Seasonality / volume check (Q4 spike, Christmas gap)
# %%
m = orders.set_index("date").resample("MS").agg(orders=("invoice", "count"), customers=("customer_id", "nunique"))
fig, ax = plt.subplots(figsize=(8, 3.5)); m.plot(ax=ax, marker="o"); ax.set_title("Monthly orders and active customers")
fig.tight_layout(); fig.savefig(OUT / "monthly_volume.png", dpi=120)
print(m)
print("\nDone. Figures + CSVs in results/eda/. Now fill the TODOs in docs/problem_statement.md")
