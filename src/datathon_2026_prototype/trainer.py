import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from .data_pipeline import load_and_process_data, prepare_model_features

def train_leak_engine(data_dir="data", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    
    # Process Data
    df = load_and_process_data(data_dir)
    X, y, feature_cols = prepare_model_features(df)

    # Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1
    }

    print("Training LightGBM Engine locally...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=250,
        valid_sets=[val_data]
    )

    auc_score = roc_auc_score(y_val, model.predict(X_val))
    print(f"✅ Model Trained Successfully. Validation AUC: {auc_score:.4f}")

    # Save Artifacts
    joblib.dump(model, f"{model_dir}/lightgbm_leak_model.pkl")
    joblib.dump(feature_cols, f"{model_dir}/feature_cols.pkl")
    print(f"Model saved to {model_dir}/lightgbm_leak_model.pkl")

if __name__ == "__main__":
    train_leak_engine()