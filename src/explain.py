"""
Generate SHAP explanations for the trained Hydrogen Jet surrogate model.

Outputs

reports/shap/

    shap_summary.png
    shap_bar.png
    feature_importance.csv
    shap_values.npy
    dependence_<feature>.png
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "final_dataset.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "YH2"
RANDOM_STATE = 42


# Load Dataset

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
X = df.drop(columns=[TARGET])

# Use a representative sample (halved to 2500)
X_sample = X.sample(min(2500, len(X)), random_state=RANDOM_STATE).reset_index(drop=True)


# Load Model


print("Loading model...")
model = joblib.load(MODEL_PATH)


# Check if model is tree-based


if not hasattr(model, "feature_importances_"):
    print("Model is not tree-based. SHAP TreeExplainer cannot be used.")
    print("Skipping SHAP analysis.")
    exit(0)


# SHAP Explainer


print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
np.save(OUTPUT_DIR / "shap_values.npy", shap_values)


# Feature Importance

importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_SHAP": np.abs(shap_values).mean(axis=0)
}).sort_values(by="Mean_SHAP", ascending=False)
importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

print("\nTop Features")
print(importance.head(10))


# SHAP Summary Plot


plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()


# SHAP Bar Plot


plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=300, bbox_inches="tight")
plt.close()


# SHAP Dependence Plots


print("Generating dependence plots...")
top_features = importance["Feature"].head(5)
for feature in top_features:
    plt.figure()
    shap.dependence_plot(feature, shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"dependence_{feature}.png", dpi=300, bbox_inches="tight")
    plt.close()


# SHAP Waterfall Plot (fixed base_value conversion)

print("Generating waterfall plot...")
sample_index = 0

# Ensure base_value is a Python float (not a numpy array)
base_value = explainer.expected_value
if hasattr(base_value, "shape"):
    if base_value.shape == ():
        base_value = base_value.item()
    elif base_value.shape == (1,):
        base_value = base_value[0]
    else:
        # For multi-output, we use the first output (should not happen)
        base_value = base_value[0] if isinstance(base_value, (list, np.ndarray)) else base_value

explanation = shap.Explanation(
    values=shap_values[sample_index],
    base_values=base_value,
    data=X_sample.iloc[sample_index],
    feature_names=X_sample.columns
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.savefig(OUTPUT_DIR / "waterfall.png", dpi=300, bbox_inches="tight")
plt.close()



print("\nSHAP analysis complete.")
print(f"Results saved to: {OUTPUT_DIR.resolve()}")