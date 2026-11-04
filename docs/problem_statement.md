# Problem statement (Stage 1 deliverable)

## Business question
An online wholesaler can send a retention offer that costs **£2** per customer. A customer who would
otherwise lapse but is retained is worth **£V** (gross margin on expected spend over the next 90 days).
**Which customers should we contact, under what budget, to maximise total expected profit?**

## Dataset
UCI Online Retail II: UK-based online retailer (many wholesale buyers), 2009-12-01 to 2011-12-09.
- Rows: TODO | Identified customers: TODO | Orders after cleaning: TODO

## Data cleaning decisions (and why)
| Decision | Why | Impact (from EDA) |
|---|---|---|
| Drop rows with no Customer ID | Cannot be tracked over time | TODO % of rows |
| Exclude cancellations (invoice starts with "C") from the order table | Returns are not purchases; kept for a return-rate feature later | TODO % |
| Exclude qty <= 0 or price <= 0 | Adjustments / bad debt / test lines | TODO % |
| Drop exact duplicate rows | The two sheets overlap in Dec 2010 | TODO rows |

## Label definition
- **Snapshot date T**: monthly (1st of month), from 2010-06-01 to 2011-09-01.
- **Active at T**: at least one purchase in the 180 days before T. Only active customers are scored.
- **Observation window**: 180 days before T (features use data strictly < T).
- **Prediction window**: 90 days. **Churn = 1 if no purchase in [T, T+90 days)**.
- **Last usable snapshot**: T + 90d must be <= 2011-12-09, so 2011-09-01 (partial windows are never labelled).

### Why these windows (fill from EDA sections 3-4)
- Inter-order gap: median TODO d, 75th pct TODO d, 90th pct TODO d => a 90d window sits at about the TODO percentile,
  so "no purchase in 90d" is a real lapse and not normal rhythm.
- Churn rate at 90d is TODO % (vs TODO % at 60d, TODO % at 120d): class balance is TODO.
- 180d observation window lets 30/60/90-day trend features exist with a full earlier comparison period.
- Business reasoning: TODO (e.g. a 90d lead time matches a quarterly buying cycle for wholesale customers).

## Time-based split plan (executed in Stage 2)
Label windows overlap in time, so splits need a purge gap.
- Train: snapshots up to 2011-03-01 (labels end by 2011-05-30)
- Validation: 2011-06-01 (labels end 2011-08-30), used for tuning AND calibration
- Test: 2011-07-01 .. 2011-09-01, touched once at the end
- Never random k-fold.

## Economics (ASSUMPTIONS, see config.yaml)
- Offer cost, gross margin, P(offer works) are assumed. No treatment history exists in this data.
- P(offer works) will be swept 0.10 - 0.40 in the Stage 7 sensitivity analysis.

## Known risks / threats to validity
- Strong Q4 seasonality: a snapshot in Sept-Oct has a different churn rate than Jan. TODO measured spread.
- Heavy-tailed revenue: top 1% of customers = TODO % of revenue. Value-weighted decisions matter.
- ~TODO % of customers have a single order: a cold-start segment with little history.
- Only ~2 years of data => few independent snapshots; keep the test set small but untouched.

## Metrics (decided now, so they can't be picked after seeing results)
ROC-AUC, PR-AUC, top-decile lift, recall at contact budget, Brier score. Not accuracy.
Baseline to beat: "no purchase in 60+ days => churn".
