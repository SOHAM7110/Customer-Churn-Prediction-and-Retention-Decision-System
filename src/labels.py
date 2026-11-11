"""Stage 2: snapshot builder, churn label, time-based split.

For each snapshot date T:
  active   = customers with >= 1 order in [T - observation_days, T)
  churn    = 1 if no order in [T, T + prediction_days)   (T inclusive, end exclusive)
Features (Stage 3) may only use orders with date < T. Labels only look at dates >= T.
"""
import pathlib
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_PARQUET = ROOT / "data/interim/snapshots.parquet"
OUT_SUMMARY = ROOT / "results/stage2_split_summary.csv"


def load_config(path=None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def snapshot_dates(cfg: dict, data_start, data_end) -> list:
    """Snapshot dates with a full observation window behind them and a fully observed label window ahead."""
    w = cfg["windows"]
    earliest = pd.Timestamp(data_start).normalize() + pd.Timedelta(days=w["observation_days"])
    first = max(pd.Timestamp(cfg["snapshots"]["first"]), earliest)
    last = pd.Timestamp(data_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(days=w["prediction_days"])
    return list(pd.date_range(first, last, freq=cfg["snapshots"]["freq"]))


def build_snapshot(orders: pd.DataFrame, T, obs_days: int, pred_days: int) -> pd.DataFrame:
    T = pd.Timestamp(T)
    if T + pd.Timedelta(days=pred_days) > orders["date"].max().normalize() + pd.Timedelta(days=1):
        raise ValueError(f"Label window for {T.date()} is not fully observed; refusing to label it.")
    hist = orders[(orders["date"] >= T - pd.Timedelta(days=obs_days)) & (orders["date"] < T)]
    fut = orders[(orders["date"] >= T) & (orders["date"] < T + pd.Timedelta(days=pred_days))]
    active = np.sort(hist["customer_id"].unique())
    returned = np.isin(active, fut["customer_id"].unique())
    return pd.DataFrame({"customer_id": active, "snapshot": T, "churn": (~returned).astype(int)})


def build_snapshots(orders: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    w = cfg["windows"]
    dates = snapshot_dates(cfg, orders["date"].min(), orders["date"].max())
    parts = [build_snapshot(orders, T, w["observation_days"], w["prediction_days"]) for T in dates]
    return pd.concat(parts, ignore_index=True)


def assign_split(snaps: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """train / val / test / purged. Purged = label window straddles a split boundary."""
    pred = pd.Timedelta(days=cfg["windows"]["prediction_days"])
    val_start = pd.Timestamp(cfg["splits"]["val_start"])
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    T = snaps["snapshot"]
    end = T + pred
    conds = [end <= val_start,
             (T >= val_start) & (end <= test_start),
             T >= test_start]
    out = snaps.copy()
    out["split"] = np.select(conds, ["train", "val", "test"], default="purged")
    return out


def check_no_leakage(snaps: pd.DataFrame, cfg: dict) -> None:
    """Every earlier split's labels must be fully known before the next split's first snapshot."""
    pred = pd.Timedelta(days=cfg["windows"]["prediction_days"])
    order = ["train", "val", "test"]
    g = snaps[snaps.split.isin(order)].groupby("split")["snapshot"].agg(["min", "max"])
    for a, b in zip(order[:-1], order[1:]):
        if a in g.index and b in g.index:
            assert g.loc[a, "max"] + pred <= g.loc[b, "min"], f"{a} label window overlaps {b} snapshots"


def split_summary(snaps: pd.DataFrame) -> pd.DataFrame:
    s = snaps.groupby("split").agg(
        snapshots=("snapshot", "nunique"), rows=("customer_id", "size"),
        customers=("customer_id", "nunique"), churn_rate=("churn", "mean"),
        first_snapshot=("snapshot", "min"), last_snapshot=("snapshot", "max")).round(3)
    return s.reindex([x for x in ["train", "val", "test", "purged"] if x in s.index])


def main():
    from src.data import load_raw, build_orders
    cfg = load_config()
    orders = build_orders(load_raw())
    snaps = assign_split(build_snapshots(orders, cfg), cfg)
    check_no_leakage(snaps, cfg)

    per_snap = (snaps.groupby(["snapshot", "split"]).agg(n_active=("customer_id", "size"),
                churn_rate=("churn", "mean")).round(3).reset_index())
    print(per_snap.to_string(index=False))
    summ = split_summary(snaps)
    print("\n", summ)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    snaps.to_parquet(OUT_PARQUET)
    summ.to_csv(OUT_SUMMARY)
    print(f"\nSaved {OUT_PARQUET} and {OUT_SUMMARY}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
