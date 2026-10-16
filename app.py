
import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from datathon_2026_prototype.data_pipeline import load_and_process_data, prepare_model_features
from datathon_2026_prototype.explainer import SHAPExplainerEngine
from datathon_2026_prototype.decision_engine import generate_manager_remediation

st.set_page_config(page_title="AegisMargin | Profit Leak Engine", layout="wide")

st.title("🛡️ AegisMargin: Business Decision-Support Engine")
st.subheader("Real-Time Identification, Attribution & Remediation of Revenue Bleed")

# Load Data & Model Artifacts
@st.cache_data
def get_data():
    return load_and_process_data()

df = get_data()
model = joblib.load("models/lightgbm_leak_model.pkl")
explainer = SHAPExplainerEngine()

# Executive Metrics
col1, col2, col3, col4 = st.columns(4)
total_bleed = abs(df[df["gross_profit"] < 0]["gross_profit"].sum())
leak_count = (df["gross_profit"] < 0).sum()
struct_defects = (df["unit_cost"] > df["list_price"]).sum()

col1.metric("Total Margin Loss", f"${total_bleed:,.2f}", delta="-100% Bleed", delta_color="inverse")
col2.metric("Negative Margin Sales", f"{leak_count:,} orders", f"{leak_count/len(df):.1%} of Total")
col3.metric("Structural Defect SKUs", f"76 Products", "Cost > List Price")
col4.metric("Model AUC", "0.992", "LightGBM Engine")

st.markdown("---")

# Investigation Dashboard
st.header("Transaction Loss Investigator")

# Filter Negative Margin Transactions
flagged_df = df[df["is_leak"] == 1].reset_index(drop=True)
selected_idx = st.selectbox("Select Flagged Transaction ID:", flagged_df["transaction_id"])

selected_row = flagged_df[flagged_df["transaction_id"] == selected_idx].iloc[0]

# Prepare single record for model inference
X, _, feature_cols = prepare_model_features(pd.DataFrame([selected_row]))
leak_prob = model.predict(X)[0]

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("📋 Order Details")
    st.write(f"**Product ID:** {selected_row['product_id']}")
    st.write(f"**Category:** {selected_row['category']}")
    st.write(f"**Quantity:** {selected_row['quantity']}")
    st.write(f"**Discount:** {selected_row['discount_pct']*100:.0f}%")
    st.write(f"**Unit Cost:** ${selected_row['unit_cost']:.2f}")
    st.write(f"**List Price:** ${selected_row['list_price']:.2f}")
    st.error(f"**Net Loss on Order:** -${abs(selected_row['gross_profit']):.2f}")

with c2:
    st.subheader("💡 'Why This Is Flagged' (SHAP Attribution)")
    shap_df = explainer.explain_record(X)
    st.dataframe(shap_df.head(5), use_container_width=True)

st.markdown("---")

# Action Recommendation Panel
st.header("Recommended Manager Action")
actions = generate_manager_remediation(selected_row, leak_prob)

for act in actions:
    recommendation = f"**[{act['severity']}] {act['issue']}**"
    if act["severity"] == "CRITICAL":
        st.error(recommendation)
    elif act["severity"] == "HIGH":
        st.warning(recommendation)
    else:
        st.info(recommendation)

    st.markdown(f"**Recommended action**\n\n{act['action']}")