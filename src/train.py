"""
Train regression models for Hydrogen Jet surrogate modeling.

Models

1. Linear Regression
2. Random Forest
3. Extra Trees
4. XGBoost

Outputs

models
    LinearRegression.pkl
    RandomForest.pkl
    ExtraTrees.pkl
    XGBoost.pkl
    best_model.pkl
    training_results.csv
"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# Configuration


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data_processed" / "final_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "YH2"
RANDOM_STATE = 42


# Load Dataset


print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]
print(f"Shape: {df.shape}")


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)


# Models (adjusted for speed)

models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,        # reduced from 300
        max_depth=12,            # reduced from 20
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1                # prints progress to console
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1             # shows progress during training
    )
}


# Training with Cross-Validation Progress


results = []
best_model = None
best_score = -np.inf

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    start_time = time.time()
    
    # Cross-validation with manual fold loop + tqdm
    
    cv_scores = []
    fold_iter = cv.split(X_train)
    total_folds = cv.get_n_splits()
    
    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(fold_iter, total=total_folds, desc=f"CV folds for {name}", unit="fold")):
        
        X_tr_fold = X_train.iloc[train_idx]
        y_tr_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        # Clone the model to avoid reusing already fitted estimators
        from sklearn.base import clone
        model_clone = clone(model)
        model_clone.fit(X_tr_fold, y_tr_fold)
        preds = model_clone.predict(X_val_fold)
        r2_fold = r2_score(y_val_fold, preds)
        cv_scores.append(r2_fold)

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    
    # Train on full training set
    
    print(f"Fitting {name} on full training set...")
    model.fit(X_train, y_train)

    # Evaluate on test set
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.1f} seconds")
    print(f"CV R²  : {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"MAE    : {mae:.4f}")
    print(f"RMSE   : {rmse:.4f}")
    print(f"R²     : {r2:.4f}")

    results.append({
        "Model": name,
        "CV_R2_Mean": cv_mean,
        "CV_R2_STD": cv_std,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Time_sec": elapsed
    })

    # Save individual model
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")

    if r2 > best_score:
        best_score = r2
        best_model = model


# Save Best Model


joblib.dump(best_model, MODEL_DIR / "best_model.pkl")


# Save Results


results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
results_df.to_csv(MODEL_DIR / "training_results.csv", index=False)

print("\n" + "=" * 60)
print("Training Summary")
print("=" * 60)
print(results_df.to_string(index=False))
print("\nBest Model")
print(results_df.iloc[0])
print(f"\nModels saved to: {MODEL_DIR.resolve()}")