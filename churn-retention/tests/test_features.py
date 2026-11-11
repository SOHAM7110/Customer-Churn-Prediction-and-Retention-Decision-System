import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src.features import build_features, FEATURES

T = "2010-06-01"
UK = "United Kingdom"

def orders_df(rows):
    d = pd.DataFrame(rows, columns=["customer_id", "date", "revenue", "n_items", "country"])
    d["date"] = pd.to_datetime(d["date"]); d["invoice"] = range(len(d))
    return d

def lines_df(rows):
    d = pd.DataFrame(rows, columns=["customer_id", "date", "stock_code", "revenue"])
    d["date"] = pd.to_datetime(d["date"]); return d

def cancels_df(rows):
    d = pd.DataFrame(rows, columns=["customer_id", "date", "revenue"])
    d["date"] = pd.to_datetime(d["date"]); return d

def close(a, b): return abs(a - b) < 1e-6

BASE_O = [(1, "2010-01-01", 200, 10, UK), (1, "2010-04-05", 50, 5, UK), (1, "2010-05-20", 150, 15, UK)]
BASE_L = [(1, "2010-04-05", "A", 50), (1, "2010-05-20", "A", 100), (1, "2010-05-20", "B", 50), (1, "2010-01-01", "C", 200)]
BASE_C = [(1, "2010-05-25", 40)]

def test_known_values():
    f = build_features(orders_df(BASE_O), lines_df(BASE_L), cancels_df(BASE_C), T, [1]).iloc[0]
    assert close(f.recency_days, 12) and f.frequency == 3 and close(f.monetary, 400)
    assert close(f.aov, 400 / 3)
    assert f.orders_30 == 1 and f.orders_60 == 2 and f.orders_90 == 2
    assert close(f.order_trend, 1 / 3)           # recent 2 vs earlier 1
    assert close(f.spend_trend, 0.0)             # recent 200 vs earlier 200
    assert close(f.mean_gap_days, 69.5) and close(f.std_gap_days, 49 / 2 ** 0.5)
    assert close(f.recency_over_gap, 12 / 69.5)
    assert close(f.avg_items_per_order, 10)
    assert f.n_unique_products == 3               # A, B, C all inside the 180d window
    assert close(f.return_rate, 40 / 400)
    assert close(f.tenure_days, 151) and f.n_orders_alltime == 3 and f.is_uk == 1

def test_old_history_affects_tenure_and_gaps_not_window_counts():
    o = orders_df([(3, "2009-11-01", 10, 1, UK), (3, "2010-05-01", 20, 2, "France")])
    f = build_features(o, lines_df([]).reindex(columns=["customer_id", "date", "stock_code", "revenue"]),
                       cancels_df([]), T, [3]).iloc[0]
    assert f.frequency == 1 and f.n_orders_alltime == 2
    assert close(f.tenure_days, 212) and close(f.mean_gap_days, 181)
    assert f.country == "France" and f.is_uk == 0

def test_future_rows_do_not_change_features():
    base = build_features(orders_df(BASE_O), lines_df(BASE_L), cancels_df(BASE_C), T, [1])
    fut_o = BASE_O + [(1, "2010-06-01", 999, 99, "Spain"),      # exactly at T: must be excluded
                      (1, "2010-06-20", 999, 99, "Spain"), (1, "2010-09-01", 999, 99, "Spain"),
                      (2, "2010-07-01", 5, 1, UK)]
    fut_l = BASE_L + [(1, "2010-06-01", "Z", 999), (1, "2010-07-01", "Y", 999)]
    fut_c = BASE_C + [(1, "2010-06-02", 500)]
    after = build_features(orders_df(fut_o), lines_df(fut_l), cancels_df(fut_c), T, [1])
    pd.testing.assert_frame_equal(base, after)
    assert set(FEATURES) <= set(after.columns)
