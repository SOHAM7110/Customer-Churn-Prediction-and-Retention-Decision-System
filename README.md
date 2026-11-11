# Churn Prediction & Retention Decision System

Predicts which customers are about to leave, turns those probabilities into a **profit-maximising contact decision**
(who, under what budget), explains each prediction, and tests the policy's value with a clearly-labelled **A/B simulation**.

**Demo:** TODO (Hugging Face Space / Streamlit Cloud link) | **Dataset:** UCI Online Retail II (about 1M transactions, 2009-12 to 2011-12)

## Business question
A retention offer costs money. A retained customer is worth money. **Whom should we contact, and under what budget?**
A prediction is only worth something if it changes an action, so the decision layer is the core of this project.

## Method at a glance
| Step | Choice | Why |
|---|---|---|
| Label | Active customer (order in last 180d); churn = no order in the next 90d, monthly snapshots | see `docs/problem_statement.md` for the evidence (gap percentiles, churn rate by window) |
| Leakage | Features use only rows dated before the snapshot; splits are time-ordered with a purge gap | `tests/test_features.py` appends future rows and proves features do not move |
| Models | Baseline rule (no purchase in 60+ days) vs logistic regression vs XGBoost vs LightGBM, light Optuna tuning | baseline must be beaten clearly |
| Imbalance | None vs class weights vs under/over-sampling, compared on validation | reported in `results/stage4_imbalance_study.csv` |
| Calibration | Platt vs isotonic vs none, chosen on a later validation snapshot, fitted on validation only | the profit maths multiplies probabilities by money |
| Explanations | SHAP (trees) / exact linear SHAP (logistic regression); correlated pairs listed | `results/stage6_*` |
| Decision | Expected profit per customer, per-customer break-even probability, top-K under a budget | `src/decide.py` |
| Validation of value | Sensitivity grid + simulated A/B with power and bootstrap CIs | `results/stage7_*` |

## Results (fill from `results/` after running the pipeline)
**Model comparison (test):** `results/stage4_test_results.csv`

| Model | ROC-AUC | PR-AUC | Top-decile lift | Recall @ 20% budget | Brier |
|---|---|---|---|---|---|
| Baseline (recency) | TODO | TODO | TODO | TODO | n/a |
| Best model | TODO | TODO | TODO | TODO | TODO |

**Calibration (test):** Brier raw TODO to calibrated TODO (`results/stage5_calibration_test.csv`, `stage5_reliability_test.png`)

**Policy comparison (backtest, ASSUMED offer effect):** `results/stage7_policy_comparison.csv`

| Policy | Contacted | Net profit | Churn loss reduced (net) |
|---|---|---|---|
| Contact nobody / everyone / default 0.5 / optimized / top-K | TODO | TODO | TODO |

**Sensitivity:** the optimized policy beats the 0.5 threshold in TODO% of assumption cells (`stage7_sensitivity.png`).
**Segments:** weakest segment TODO (`results/stage5_segments_test.csv`).

## Caveats (read these)
- **The offer success rate is an assumption.** The data has no treatment history, so P(offer works), discount, margin and
  contact cost are inputs, not measurements. Every profit number is conditional on them. The sensitivity grid shows how
  far the conclusions hold.
- **The A/B result is a simulation.** It proves the test design (power, sample size, bootstrap CI) works and shows how large a
  real experiment must be. It does not prove the offer works.
- **Calibration can drift.** Validation and test are different periods (the test labels cover Q4 buying season). If the
  churn base rate shifts, probabilities are biased, and a profit-optimised threshold is more sensitive to that than a
  default 0.5 rule. Check `stage5_reliability_test.png` and say so if the optimized policy does not beat the default.
- **Correlated features dilute SHAP importance** (see `results/stage6_correlated_features.csv`). Read importance per
  group of related features, not per column.
- **Few independent snapshots.** About two years of data gives a small validation and test set; rows from the same
  customer are correlated. Treat small differences between models as noise.
- **Customers without an ID are dropped** (about a fifth of rows in this dataset), and tenure is left-censored at the data start.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make data labels features baseline train      # stages 0-4
make calibrate explain                         # stages 5-6, uses validation only
make final                                     # test split used ONLY for reporting from here on
make export && make app                        # stage 8
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Order matters for honesty: everything that selects or tunes (models, imbalance strategy, hyper-parameters, calibration
method) happens on validation. `make final` then reports on the untouched test split, and `train --final-test` refuses to run twice.

## Repo map
```
config.yaml           windows, splits, economics (all assumptions in one place)
src/labels.py         snapshots, churn label, purged time split
src/features.py       leakage-safe features (22) | src/metrics.py shared metrics
src/models.py         factories, imbalance strategies, Optuna spaces
src/train.py          comparison + MLflow | src/baseline.py rule baseline
src/calibrate.py      calibration + segment checks | src/explain.py SHAP
src/decide.py         expected profit, policies, sensitivity, budget curve
src/simulate_ab.py    A/B simulation | src/export_app_data.py app bundle
app/streamlit_app.py  demo | app/api.py optional FastAPI /predict
docs/                 problem statement, feature dictionary, deploy guide, resume + interview mapping
tests/                leakage, label, metric, calibration, decision-maths and A/B tests
```
