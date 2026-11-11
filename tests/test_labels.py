import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from src.labels import build_snapshot, build_snapshots, assign_split, check_no_leakage, snapshot_dates

def mk(rows):
    return pd.DataFrame(rows, columns=["customer_id", "date"]).assign(date=lambda d: pd.to_datetime(d.date))

T = "2010-03-01"

def test_label_and_activity_rules():
    orders = mk([
        ("A", "2010-01-10"), ("A", "2010-03-15"),   # active, returns inside window -> churn 0
        ("B", "2010-01-10"),                        # active, never returns       -> churn 1
        ("C", "2010-04-01"),                        # first order after T         -> not scored
        ("D", "2010-03-01"),                        # order AT T is future, not history -> not scored
        ("E", "2009-08-01"),                        # older than 180d             -> not scored
        ("F", "2010-06-30"),                        # keeps data_end late enough
    ])
    # use numeric-like ids
    orders["customer_id"] = orders.customer_id.map({k: i for i, k in enumerate("ABCDEF")})
    snap = build_snapshot(orders, T, 180, 90).set_index("customer_id").churn
    assert snap.to_dict() == {0: 0, 1: 1}

def test_window_end_is_exclusive():
    orders = mk([(1, "2010-01-05"), (1, "2010-05-30")])   # 2010-03-01 + 90d = 2010-05-30
    assert build_snapshot(orders, T, 180, 90).churn.iloc[0] == 1

def test_refuses_unobserved_label_window():
    orders = mk([(1, "2010-01-05"), (1, "2010-04-01")])
    with pytest.raises(ValueError):
        build_snapshot(orders, T, 180, 90)

def test_split_purges_boundary_snapshots():
    cfg = {"windows": {"observation_days": 180, "prediction_days": 90},
           "splits": {"val_start": "2011-04-01", "test_start": "2011-08-01"}}
    d = pd.date_range("2010-06-01", "2011-09-01", freq="MS")
    snaps = assign_split(pd.DataFrame({"customer_id": 1, "snapshot": d, "churn": 0}), cfg)
    by = snaps.groupby("split").snapshot.apply(lambda s: [x.strftime("%Y-%m") for x in s]).to_dict()
    assert by["train"][0] == "2010-06" and by["train"][-1] == "2011-01"
    assert by["val"] == ["2011-04", "2011-05"]
    assert by["test"] == ["2011-08", "2011-09"]
    assert by["purged"] == ["2011-02", "2011-03", "2011-06", "2011-07"]
    check_no_leakage(snaps, cfg)

def test_snapshot_dates_respect_bounds():
    cfg = {"windows": {"observation_days": 180, "prediction_days": 90},
           "snapshots": {"first": "2010-06-01", "freq": "MS"}}
    ds = snapshot_dates(cfg, "2009-12-01", "2011-12-09 17:00")
    assert ds[0] == pd.Timestamp("2010-06-01") and ds[-1] == pd.Timestamp("2011-09-01")
