import pandas as pd
import numpy as np

def load_and_process_data(data_dir="data"):
    # Loads the required datasets and performs initial processing.
    transactions = pd.read_csv(f"{data_dir}/transactions.csv")
    products = pd.read_csv(f"{data_dir}/products.csv")
    customers = pd.read_csv(f"{data_dir}/customers.csv")

    # Merge Core Datasets
    df = transactions.merge(products, on="product_id", how="left")
    df = df.merge(customers, on="customer_id", how="left")

    # Feature Engineering
    df["effective_revenue"] = df["quantity"] * df["unit_price"] * (1 - df["discount_pct"])
    df["cogs"] = df["quantity"] * df["unit_cost"]
    df["gross_profit"] = df["effective_revenue"] - df["cogs"]
    df["cost_price_ratio"] = df["unit_cost"] / df["list_price"].replace(0, 1)
    df["unit_cost_delta"] = df["unit_cost"] - df["list_price"]

    # Target Labeling: 1 if Gross Profit < 0 (Money Leak), else 0
    df["is_leak"] = (df["gross_profit"] < 0).astype(int)

    return df

def prepare_model_features(df):
    # returns paramters for model training
    feature_cols = [
        "quantity", "unit_price", "discount_pct", 
        "unit_cost", "list_price", "supplier_lead_days", 
        "cost_price_ratio", "unit_cost_delta"
    ]
    
    X = df[feature_cols]
    y = df["is_leak"]
    return X, y, feature_cols