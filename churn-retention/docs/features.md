# Feature dictionary (Stage 3)

All features at snapshot T use only rows with date < T. Windows are 180 days unless stated.

| Feature | Definition | Leakage note |
|---|---|---|
| recency_days | days from last order before T to T | |
| frequency | orders in 180d window | |
| monetary | revenue in 180d window | |
| aov | monetary / frequency | |
| orders_30/60/90, revenue_30/60/90 | activity in the last 30/60/90 days | |
| order_trend, spend_trend | (last 90d - earlier 90d) / (sum), range -1..1 | |
| mean_gap_days, std_gap_days, gap_cv | inter-order gap stats over ALL history before T | std is NaN until 3 orders |
| recency_over_gap | recency_days / mean_gap_days (>1 = overdue vs own rhythm) | NaN for one-order customers |
| avg_items_per_order | items per order in window | |
| n_unique_products | distinct stock codes in window | |
| return_rate | cancelled value / purchased value in window | cancellations are dated by the cancel invoice |
| tenure_days, n_orders_alltime | since first order before T | left-censored: data starts 2009-12 |
| is_uk | last known country == UK | |

Deliberately excluded: month / day-of-year. With about two years of data a seasonal feature would let a model
memorise the calendar instead of behaviour. Country is kept as a metadata column for Stage 5 segment checks.

Missing values (single-order customers) are left as NaN: XGBoost/LightGBM handle them natively. For logistic
regression (Stage 4) impute the median and add a missing-indicator inside the sklearn pipeline.
