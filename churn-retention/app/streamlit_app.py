"""Churn retention decision app.   streamlit run app/streamlit_app.py"""
import json
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decide import budget_curve, econ, policy_table, unit_profit
from src.explain import top_reasons
from src.features import FEATURES
from src.labels import load_config

st.set_page_config(page_title="Churn retention decisions", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet(ROOT / "app/data/customers.parquet")
    return df, json.loads((ROOT / "app/data/meta.json").read_text())


if not (ROOT / "app/data/customers.parquet").exists():
    st.error("App data missing. Run `python -m src.export_app_data` first.")
    st.stop()

df, meta = load()
cfg = load_config()
e0 = econ(cfg)

st.title("Customer churn: who should we contact?")
st.caption(f"Model: {meta['model']} | probabilities calibrated with: {meta['calibration']} | data: held-out test period "
           "(UCI Online Retail II). **The offer success rate is an assumption, not a measured value.**")

sb = st.sidebar
sb.header("Assumptions (all adjustable)")
w = sb.slider("P(offer works): ASSUMED", 0.05, 0.60, float(e0["w"]), 0.01)
d = sb.slider("Discount paid on next-90d revenue", 0.00, 0.20, float(e0["d"]), 0.005)
m = sb.slider("Gross margin", 0.10, 0.60, float(e0["m"]), 0.01)
c = sb.slider("Fixed cost per contact (GBP)", 0.0, 10.0, float(e0["c"]), 0.5)
budget = sb.slider("Contact budget (% of customers)", 0, 100, int(cfg["budget"]["contact_fraction"] * 100))
snaps = sorted(df.snapshot.unique())
snap = sb.selectbox("Snapshot date", snaps, index=len(snaps) - 1, format_func=lambda t: str(pd.Timestamp(t).date()))

e = {"w": w, "m": m, "d": d, "c": c}
cur = df[df.snapshot == snap].reset_index(drop=True)
p, R, y = cur.p_cal.values, cur.R90.values, cur.churn.values
profit = unit_profit(p, R, e)

tab1, tab2, tab3 = st.tabs(["Customer lookup", "Budget and profit", "About and caveats"])

with tab1:
    order = cur.sort_values("p_cal", ascending=False).customer_id.tolist()
    cid = st.selectbox("Customer ID (sorted by risk, highest first)", order)
    i = int(cur.index[cur.customer_id == cid][0])
    row = cur.iloc[i]
    k = st.columns(4)
    k[0].metric("Churn probability (calibrated)", f"{row.p_cal:.1%}")
    k[1].metric("Risk percentile", f"{(p < row.p_cal).mean():.0%}")
    k[2].metric("Expected 90d revenue", f"GBP {row.R90:,.0f}")
    k[3].metric("Expected profit if contacted", f"GBP {profit[i]:,.1f}")
    if profit[i] > 0:
        st.success(f"Recommended action: CONTACT. Break-even churn probability for this customer is "
                   f"{(e['c'] / max(row.R90, 1e-9) + e['d']) / (e['w'] * e['m']):.0%}.")
    else:
        st.info("Recommended action: DO NOT contact (expected profit is not positive under the current assumptions).")
    sv = pd.Series({f: row[f"shap__{f}"] for f in FEATURES})
    reasons = top_reasons(sv, row[FEATURES], 6)
    st.subheader("Why is this customer at risk?")
    for r in reasons:
        st.write(("🔺 " if r["shap"] > 0 else "🔻 ") + r["text"])
    fig, ax = plt.subplots(figsize=(6, 2.8))
    rr = reasons[::-1]
    ax.barh([x["label"] for x in rr], [x["shap"] for x in rr], color=["tab:red" if x["shap"] > 0 else "tab:blue" for x in rr])
    ax.axvline(0, color="grey"); ax.set_xlabel("contribution to churn log-odds (model, before calibration)")
    fig.tight_layout(); st.pyplot(fig)
    if "churn" in row:
        st.caption(f"Backtest only: this customer {'did' if row.churn == 1 else 'did not'} actually lapse in the following 90 days.")

with tab2:
    bc = budget_curve(p, R, y, e)
    at = bc.iloc[int(budget)]
    j = st.columns(4)
    j[0].metric("Customers contacted", f"{int(at.contacted):,}")
    j[1].metric("Expected profit (model)", f"GBP {at.expected_profit_model:,.0f}")
    j[2].metric("Backtest profit (actual churn)", f"GBP {at.backtest_profit:,.0f}")
    j[3].metric("Optimal budget", f"{bc.budget_frac[bc.expected_profit_model.idxmax()]:.0%}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(bc.budget_frac * 100, bc.expected_profit_model, label="expected (model probabilities)")
    ax.plot(bc.budget_frac * 100, bc.backtest_profit, label="backtest (actual churn, assumed offer effect)")
    ax.axvline(budget, color="grey", ls="--"); ax.set(xlabel="% of customers contacted (best first)", ylabel="net profit, GBP"); ax.legend()
    fig.tight_layout(); st.pyplot(fig)
    st.subheader("Policy comparison under the current assumptions (backtest)")
    pt = policy_table(p, cur.p_raw.values, y, R, e, budget / 100 if budget else 0.0)
    st.dataframe(pt[["policy", "contacted", "contact_rate", "precision", "net_profit", "net_loss_reduction_pct"]].round(3))

with tab3:
    st.markdown("""
**What is measured vs assumed.** Churn probabilities, ranking quality and calibration are measured on held-out later periods.
The offer success rate, discount, margin and contact cost are **assumptions**: this dataset has no treatment history.
A real randomised test is needed before trusting any profit number here.
""")
    for name, title in [("stage4_model_comparison.csv", "Model comparison (validation)"),
                        ("stage4_test_results.csv", "Model comparison (test)"),
                        ("stage7_ab_simulation.csv", "A/B simulation summary (SIMULATION)")]:
        f = ROOT / "results" / name
        if f.exists():
            st.subheader(title); st.dataframe(pd.read_csv(f))
    for name, cap in [("stage5_reliability_test.png", "Reliability curve (test)"), ("stage5_segments_test.png", "Segment checks (test)"),
                      ("stage6_shap_global.png", "What drives churn"), ("stage7_sensitivity.png", "Sensitivity of the policy gain")]:
        f = ROOT / "results" / name
        if f.exists():
            st.image(str(f), caption=cap)
