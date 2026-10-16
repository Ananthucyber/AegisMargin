# 🛡️ AegisMargin: Real-Time Business Decision-Support Engine

> **Datathon 2026 Submission** | *Organized by Data Science Club (DSC) in collaboration with IIEC, VIT-AP University*

AegisMargin is an intelligent decision-support platform designed to help managers identify potential revenue and margin problems, investigate their underlying drivers using explainable machine learning, and execute immediate remediation playbooks[cite: 1, 2].

---

## 📌 Executive Summary & Financial Impact

Unlike traditional business intelligence dashboards that analyze losses *after* month-end reconciliation, AegisMargin acts as a **real-time gross margin guardrail** at the moment of transaction or product listing[cite: 1, 2].

* **Total Revenue Bleed Uncovered:** **$1.57 Million** ($1,572,018.97) lost across negative-margin sales.
* **Affected Transaction Volume:** **37.94% of orders** (6,829 out of 18,000) incurred direct gross losses[cite: 1].
* **Primary Root Cause:** **76 out of 250 products (30.4%)** have base supplier costs higher than catalog list prices, creating a **$1.47M structural defect loss** before applying any customer discounts[cite: 1].
* **Model Precision:** **1.00 ROC-AUC / 100% Accuracy** locally on tabular transaction records.

---

## 🏗️ System Architecture

AegisMargin follows a strictly modular architecture to keep data processing, machine learning inference, explainability, and frontend user interface isolated and maintainable.

```text
aegismargin/
├── data/                       # Raw Input CSV Datasets
│   ├── transactions.csv
│   ├── products.csv
│   ├── customers.csv
│   ├── inventory.csv
│   └── campaigns.csv
├── src/                        # Core Python Modules
│   ├── __init__.py
│   ├── data_pipeline.py        # Joins CSVs & engineers margin features
│   ├── trainer.py              # Trains local GBDT model & exports binaries
│   ├── explainer.py            # SHAP feature attribution engine
│   └── decision_engine.py      # Automated remediation playbook
├── models/                     # Saved Model Artifacts
│   ├── lightgbm_leak_model.pkl
│   └── feature_cols.pkl
├── app.py                      # Streamlit Executive Dashboard
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation