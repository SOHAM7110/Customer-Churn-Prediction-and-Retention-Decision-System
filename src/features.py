"""Stage 3: leakage-safe features.

RULE: every feature at snapshot T is computed from rows with date < T. The slicing happens once, at the top of
build_features(), so nothing downstream can see the future. tests/test_features.py proves it by appending
future rows and checking the features do not move.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPS = ROOT / "data/interim/snapshots.parquet"
OUT = ROOT / "data/interim/features.parquet"
DAY = pd.Timedelta(days=1)

FEATURES = [
    # RFM over the 180-day window
    "recency_days", "frequency", "monetary", "aov",
    # short-horizon activity
    "orders_30", "orders_60", "orders_90", "revenue_30", "revenue_60", "revenue_90",
    # trend: last 90d vs the 90d before (bounded -1..1; -1 = all activity was in the earlier half)
    "order_trend", "spend_trend",
    # rhythm (all history before T)
    "mean_gap_days", "std_gap_days", "gap_cv", "recency_over_gap",
    # behaviour
    "avg_items_per_order", "n_unique_products", "return_rate",
    # profile
    "tenure_days", "n_orders_alltime", "is_uk",
]
META = ["customer_id", "snapshot", "country", "churn", "split"]   # country kept for Stage 5 segment checks


def prepare_inputs(raw: pd.DataFrame):
    """Raw lines -> (orders, purchase lines, cancellation lines), all for identified customers."""
    from src.data import build_orders
    orders = build_orders(raw)
    has_id = raw.customer_id.notna()
    ok = (~raw.is_cancel) & (raw.quantity > 0) & (raw.price > 0) & has_id
    lines = raw.loc[ok, ["customer_id", "invoice_date", "stock_code", "line_revenue"]].rename(
        columns={"invoice_date": "date", "line_revenue": "revenue"})
    lines["customer_id"] = lines.customer_id.astype(int)
    cx = raw.loc[raw.is_cancel & has_id, ["customer_id", "invoice_date", "line_revenue"]].rename(
        columns={"invoice_date": "date", "line_revenue": "revenue"})
    cx["customer_id"] = cx.customer_id.astype(int)
    cx["revenue"] = cx["revenue"].abs()
    return orders, lines, cx


def build_features(orders, lines, cancels, T, ids, obs_days: int = 180) -> pd.DataFrame:
    assert obs_days == 180, "trend features assume a 180-day window (90 recent vs 90 earlier)"
    T = pd.Timestamp(T)
    ids = pd.Index(ids, name="customer_id")
    start = T - obs_days * DAY

    # ---- the ONLY place data enters: everything is sliced strictly before T ----
    o_all = orders[(orders.date < T) & orders.customer_id.isin(ids)].sort_values(["customer_id", "date"])
    o_w = o_all[o_all.date >= start]
    l_w = lines[(lines.date >= start) & (lines.date < T) & lines.customer_id.isin(ids)]
    c_w = cancels[(cancels.date >= start) & (cancels.date < T) & cancels.customer_id.isin(ids)]
    # -----------------------------------------------------------------------------

    def cnt(d):
        return d.groupby("customer_id").size().reindex(ids, fill_value=0)

    def tot(d, col):
        return d.groupby("customer_id")[col].sum().reindex(ids, fill_value=0.0)

    def safe(den):                       # avoid divide-by-zero -> NaN instead of inf
        return den.where(den > 0)

    f = pd.DataFrame(index=ids)
    g = o_all.groupby("customer_id")
    last, first = g.date.max().reindex(ids), g.date.min().reindex(ids)

    f["recency_days"] = (T - last).dt.total_seconds() / 86400
    f["frequency"] = cnt(o_w)
    f["monetary"] = tot(o_w, "revenue")
    f["aov"] = f["monetary"] / safe(f["frequency"])

    for d in (30, 60, 90):
        w = o_w[o_w.date >= T - d * DAY]
        f[f"orders_{d}"] = cnt(w)
        f[f"revenue_{d}"] = tot(w, "revenue")

    earlier = o_w[o_w.date < T - 90 * DAY]
    o_rec, o_ear = f["orders_90"], cnt(earlier)
    s_rec, s_ear = f["revenue_90"], tot(earlier, "revenue")
    f["order_trend"] = (o_rec - o_ear) / safe(o_rec + o_ear)
    f["spend_trend"] = (s_rec - s_ear) / safe(s_rec + s_ear)

    gap = o_all.groupby("customer_id").date.diff().dt.total_seconds() / 86400
    gs = gap.groupby(o_all["customer_id"]).agg(["mean", "std"]).reindex(ids)
    f["mean_gap_days"] = gs["mean"]
    f["std_gap_days"] = gs["std"]                       # NaN until a customer has >= 3 orders
    f["gap_cv"] = gs["std"] / safe(gs["mean"])
    f["recency_over_gap"] = f["recency_days"] / safe(gs["mean"])   # >1 means "overdue" vs own rhythm

    f["avg_items_per_order"] = tot(o_w, "n_items") / safe(f["frequency"])
    f["n_unique_products"] = l_w.groupby("customer_id").stock_code.nunique().reindex(ids, fill_value=0)
    f["return_rate"] = tot(c_w, "revenue") / safe(f["monetary"])

    f["tenure_days"] = (T - first).dt.total_seconds() / 86400     # left-censored at the data start (2009-12)
    f["n_orders_alltime"] = cnt(o_all)
    f["country"] = g.country.last().reindex(ids)                   # last country seen before T
    f["is_uk"] = (f["country"] == "United Kingdom").astype(int)
    return f[FEATURES + ["country"]].reset_index()


def build_feature_table(snaps, orders, lines, cancels, obs_days: int = 180) -> pd.DataFrame:
    snaps = snaps[snaps.split != "purged"]               # purged snapshots are never modelled
    parts = []
    for T, sub in snaps.groupby("snapshot"):
        f = build_features(orders, lines, cancels, T, sub.customer_id.values, obs_days)
        f["snapshot"] = T
        f["churn"] = sub.churn.values                    # same order as ids
        f["split"] = sub.split.values
        parts.append(f)
    return pd.concat(parts, ignore_index=True)[META[:2] + FEATURES + META[2:]]


def main():
    from sklearn.metrics import roc_auc_score
    from src.data import load_raw
    if not SNAPS.exists():
        raise SystemExit("Run `python -m src.labels` first (Stage 2).")
    orders, lines, cancels = prepare_inputs(load_raw())
    table = build_feature_table(pd.read_parquet(SNAPS), orders, lines, cancels)

    # ---- sanity checks ----
    assert not np.isinf(table[FEATURES].to_numpy(dtype=float)).any(), "inf found"
    assert (table.recency_days >= 0).all() and (table.frequency >= 1).all() and (table.monetary > 0).all()
    table.to_parquet(OUT)
    print(f"features: {table.shape}  ->  {OUT}\n")

    tr = table[table.split == "train"]
    rep = pd.DataFrame({"nan_%": (table[FEATURES].isna().mean() * 100).round(1)})
    rep["train_univariate_auc"] = [
        round(roc_auc_score(tr.churn[tr[c].notna()], tr[c].dropna()), 3) for c in FEATURES]
    print(rep.to_string())
    print("\nAUC > 0.5: higher value => more churn. recency_days should be well above 0.5; "
          "frequency/orders_* should be below 0.5. If not, suspect a bug.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
