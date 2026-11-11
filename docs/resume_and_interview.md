# Where each resume number comes from

| Resume placeholder | Source file | Column / row |
|---|---|---|
| Dataset, N customers | `results/eda/data_quality.csv`, `results/stage2_split_summary.csv` | rows, customers |
| ROC-AUC, PR-AUC, baseline comparison | `results/stage4_test_results.csv` | roc_auc, pr_auc, pr_auc_gain_vs_baseline |
| Brier score | `results/stage5_calibration_test.csv` | brier, raw vs calibrated |
| "Simulated churn loss reduced by X%" | `results/stage7_policy_comparison.csv` | net_loss_reduction_pct, optimized vs default threshold |
| Robustness | `results/stage7_sensitivity.csv` | share of cells where the policy wins |
| A/B design | `results/stage7_ab_simulation.csv` | n_per_arm_for_target_power, boot CI |

Write the policy number honestly: "simulated, assuming X% offer success", and use the default-0.5 comparison only if
your optimized policy actually wins on your data. If it does not, that is a finding: explain it with the reliability plot.

# Interview prompts mapped to the repo
- Why that churn window? `docs/problem_statement.md` (gap percentiles, churn rate by window).
- How did you prevent leakage? `src/features.py` single slice at T + `tests/test_features.py`; purged splits in `src/labels.py`.
- Why PR-AUC, not accuracy? `src/metrics.py`; churn is a minority or shifting class.
- Why calibration? `src/decide.py`: profit multiplies probability by money. See the "UNcalibrated probs" row in the policy table.
- How would you validate the offer for real? `src/simulate_ab.py`: randomise, measure margin per customer, size for power; state what the simulation does not prove.
- What changes in production? Drift check on key features, retraining cadence, logging contact outcomes to replace the assumed offer effect.
