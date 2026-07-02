""""
Exploratory Data Analysis

Outputs

reports/figures/
    dataset_summary.csv
    missing_values.csv
    feature_statistics.csv
    correlation_heatmap.png
    target_distribution.png
    feature_distributions.png
    spatial_distribution.png
    velocity_vs_target.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "final_dataset.parquet"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "YH2"
sns.set_theme(style="whitegrid")


# Load Dataset

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Dataset Shape : {df.shape}")


# Dataset Summary


print("\nDataset Information")
df.info()

summary = pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100,
    "unique": df.nunique()
})
summary.to_csv(FIGURE_DIR / "dataset_summary.csv")


# Missing Values


missing = pd.DataFrame({
    "Missing Values": df.isna().sum(),
    "Percentage": df.isna().mean() * 100
})
missing.to_csv(FIGURE_DIR / "missing_values.csv")

stats = df.describe().T
stats.to_csv(FIGURE_DIR / "feature_statistics.csv")


# Corr Heatmap


plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


# Target Distribution

plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], bins=60, kde=True)
plt.title(f"{TARGET} Distribution")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "target_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# Feature Distributions

features = [
    "RHO_kgm-3", "UX_ms-1", "UY_ms-1", "P_Pa", "T_K",
    "velocity_mag", "radius"
]
ncols = 2
nrows = (len(features) + 1) // 2
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
axes = axes.flatten()
for ax, feature in zip(axes, features):
    sns.histplot(df[feature], bins=50, kde=True, ax=ax)
    ax.set_title(feature)
for ax in axes[len(features):]:
    fig.delaxes(ax)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "feature_distributions.png", dpi=300, bbox_inches="tight")
plt.close()


# Spatial Distribution

sample = df.sample(min(5000, len(df)), random_state=42)
plt.figure(figsize=(8, 6))
plt.scatter(sample["X"], sample["Y"], c=sample[TARGET], s=2, cmap="viridis")
plt.colorbar(label=TARGET)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Spatial Distribution of Hydrogen Mass Fraction")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "spatial_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# Velocity vs Hydrogen

plt.figure(figsize=(8, 6))
sns.scatterplot(data=sample, x="velocity_mag", y=TARGET, s=10, alpha=0.5)
plt.title("Velocity Magnitude vs Hydrogen Mass Fraction")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "velocity_vs_target.png", dpi=300, bbox_inches="tight")
plt.close()


# Top Correlated Features


print("\nTop Correlations With Target\n")
corr_target = corr[TARGET].sort_values(ascending=False)
print(corr_target)
corr_target.to_csv(FIGURE_DIR / "target_correlations.csv")

print("\nEDA Complete")
print(f"Figures saved to: {FIGURE_DIR.resolve()}")