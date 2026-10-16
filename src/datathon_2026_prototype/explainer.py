import joblib
import shap
import pandas as pd

class SHAPExplainerEngine:
    def __init__(self, model_path="models/lightgbm_leak_model.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_record(self, record_df):
        """
        Calculates top feature drivers pushing a record into leak territory.
        """
        shap_values = self.explainer.shap_values(record_df)
        
        # Handle binary classification outputs
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        impact = pd.DataFrame({
            "Feature": record_df.columns,
            "Value": record_df.iloc[0].values,
            "SHAP_Impact": vals
        }).sort_values(by="SHAP_Impact", key=abs, ascending=False)

        return impact