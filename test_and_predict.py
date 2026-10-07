"""
ML Assignment 1 - Polynomial Regression
Roll Number: BT2024105

Models:
  Var1: Net Power Score prediction using all 6 operational parameters (Degree 3 Polynomial, Ridge alpha=1e-4)
  Var2: Thermal Anomaly Score prediction using all 3 spatial coordinates (Degree 4 Polynomial, Ridge alpha=1e-4)

Deliverables Generated in Workspace:
  - BT2024105_pred_var1.csv
  - BT2024105_pred_var2.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

ROLL_NO = "BT2024105"
np.random.seed(42)
KF = KFold(n_splits=5, shuffle=True, random_state=42)

def build_pipeline(degree: int, alpha: float = 1e-4) -> Pipeline:
    """Build a Polynomial + StandardScaler + Ridge pipeline."""
    return Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=True)),
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha))
    ])

def evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, label: str = ""):
    """Print train and 5-fold CV metrics."""
    y_hat = pipeline.predict(X)
    tr_mse = mean_squared_error(y, y_hat)
    tr_r2  = r2_score(y, y_hat)

    cv_r2  = cross_val_score(pipeline, X, y, cv=KF, scoring="r2")
    cv_mse = -cross_val_score(pipeline, X, y, cv=KF, scoring="neg_mean_squared_error")

    print(f"  [{label}]")
    print(f"    Train  MSE={tr_mse:.5f}   R2={tr_r2:.5f}")
    print(f"    CV     MSE={cv_mse.mean():.5f} +/- {cv_mse.std():.5f}")
    print(f"    CV     R2 ={cv_r2.mean():.5f} +/- {cv_r2.std():.5f}")

def save_predictions(y_pred: np.ndarray, var_id: str):
    """Save predictions strictly to the official naming format: <ROLLNO>_pred_var<ID>.csv"""
    filename = f"{ROLL_NO}_pred_{var_id}.csv"
    pd.DataFrame({"y": y_pred}).to_csv(filename, index=False)
    print(f"  Saved -> {filename} ({len(y_pred)} rows)")

def solve_var1():
    print("\n" + "=" * 60)
    print("VAR 1 : Steam Turbine Optimization (Degree 3, All 6 Features)")
    print("=" * 60)

    train_path = f"trainAndTestData/{ROLL_NO}_train_var1.csv"
    test_path  = f"trainAndTestData/{ROLL_NO}_test_var1.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    features = ["x1", "x2", "x3", "x4", "x5", "x6"]
    X_train  = train_df[features].values
    y_train  = train_df["y"].values
    X_test   = test_df[features].values

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    pipe = build_pipeline(degree=3, alpha=1e-4)
    pipe.fit(X_train, y_train)
    evaluate(pipe, X_train, y_train, label="var1: Deg 3 Ridge (All 6 Features)")

    y_pred = pipe.predict(X_test)
    save_predictions(y_pred, "var1")
    return pipe

def solve_var2():
    print("\n" + "=" * 60)
    print("VAR 2 : Subterranean Thermal Mapping (Degree 4, All 3 Coordinates)")
    print("=" * 60)

    train_path = f"trainAndTestData/{ROLL_NO}_train_var2.csv"
    test_path  = f"trainAndTestData/{ROLL_NO}_test_var2.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Prompt guidance check: Model A (x1 only, deg 4) vs Model B (all 3 coords, deg 4)
    X_train_x1 = train_df[["x1"]].values
    pipe_x1 = build_pipeline(degree=4, alpha=1e-4)
    pipe_x1.fit(X_train_x1, train_df["y"].values)
    print("Prompt Guidance Ablation Check (Model A: x1 only, Degree 4):")
    evaluate(pipe_x1, X_train_x1, train_df["y"].values, label="Model A (x1 only, deg 4)")

    # Model B: All 3 coordinates
    features = ["x1", "x2", "x3"]
    X_train  = train_df[features].values
    y_train  = train_df["y"].values
    X_test   = test_df[features].values

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    pipe = build_pipeline(degree=4, alpha=1e-4)
    pipe.fit(X_train, y_train)
    print("Selected Optimal Model (Model B: All 3 coordinates, Degree 4):")
    evaluate(pipe, X_train, y_train, label="Model B (x1, x2, x3, deg 4)")

    y_pred = pipe.predict(X_test)
    save_predictions(y_pred, "var2")
    return pipe

if __name__ == "__main__":
    solve_var1()
    solve_var2()
    print("\n[SUCCESS] All pipeline training and inference finished successfully.")
