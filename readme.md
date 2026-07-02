# Machine Learning Surrogate Model for Hydrogen Jet Dispersion Prediction

# Overview

This project develops a machine learning surrogate model for predicting hydrogen mass fraction from high-fidelity Computational Fluid Dynamics (CFD) simulations of a lifted hydrogen jet flame.

Instead of repeatedly running computationally expensive CFD simulations, the trained machine learning model predicts the hydrogen concentration field using local flow properties, providing a fast approximation suitable for engineering analysis.

The project follows a complete end-to-end machine learning workflow including data processing, exploratory data analysis (EDA), model training, evaluation, and model explainability.


# Project Objectives

 Process large-scale CFD simulation data
 Build a tabular machine learning dataset
 Train regression models for hydrogen concentration prediction
 Compare multiple machine learning algorithms
 Evaluate model performance
 Interpret predictions using SHAP explainability


# Target Variable

Hydrogen Mass Fraction


# Machine Learning Workflow


Raw CFD Data
        │
        ▼
process_data.py
        │
        ▼
Processed Dataset
(final_dataset.parquet)
        │
        ▼
eda.py
        │
        ▼
train.py
        │
        ▼
Best Model
        │
        ▼
evaluate.py
        │
        ▼
Performance Report
        │
        ▼
explain.py



# Models

The following regression models are trained and compared:

  Linear Regression
  Random Forest
  Extra Trees
  XGBoost

Models are evaluated using 5-fold cross-validation.


# Evaluation Metrics

  Mean Absolute Error (MAE)
  Root Mean Squared Error (RMSE)
  Coefficient of Determination (R²)
