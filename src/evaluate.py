"""
Evaluate the trained regression model.

Outputs

reports/figures/
    predicted_vs_actual.png
    residual_plot.png
    residual_distribution.png
    feature_importance.png

reports/
    evaluation_metrics.csv
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "final_dataset.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "YH2"
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# Load Dataset

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)


# Load Model

print("Loading trained model...")
model = joblib.load(MODEL_PATH)


# Prediction

predictions = model.predict(X_test)


# Metrics

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

metrics = pd.DataFrame({"Metric": ["MAE", "RMSE", "R2"], "Value": [mae, rmse, r2]})
metrics.to_csv(REPORT_DIR / "evaluation_metrics.csv", index=False)

print("\nEvaluation Metrics")
print(metrics)


# Predicted vs Actual

plt.figure(figsize=(7, 7))
plt.scatter(y_test, predictions, alpha=0.4, s=10)
mn = min(y_test.min(), predictions.min())
mx = max(y_test.max(), predictions.max())
plt.plot([mn, mx], [mn, mx], "r--", linewidth=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "predicted_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()


# Residual Plot

residuals = y_test - predictions
plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals, alpha=0.4, s=10)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "residual_plot.png", dpi=300, bbox_inches="tight")
plt.close()


# Residual Distribution


plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "residual_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# Feature Importance

if hasattr(model, "feature_importances_"):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance.head(15), x="Importance", y="Feature")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


# Finish

print("\nEvaluation Complete")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"\nReports saved to: {REPORT_DIR.resolve()}")