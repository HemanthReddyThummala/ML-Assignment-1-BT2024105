"""
Step 4: Regularization Strength (Alpha) Tuning
===============================================
After locking in the best feature sets and degrees:
  - Var1: All 6 features, Degree 3
  - Var2: All 3 features, Degree 4

This script tunes the Ridge regularization parameter alpha
across a log-spaced grid: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
and reports 5-Fold CV MSE and CV R² for each value.

Conclusion: alpha = 1e-4 provides optimal numerical stability 
without degrading cross-validation performance.

Run: python 04_alpha_tuning.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

ROLL_NO = "BT2024105"
DATA_DIR = "../trainAndTestData"

train1 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var1.csv")
train2 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var2.csv")

X1 = train1[["x1", "x2", "x3", "x4", "x5", "x6"]].values
y1 = train1["y"].values

X2 = train2[["x1", "x2", "x3"]].values
y2 = train2["y"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]

def tune_alpha(X, y, degree, label):
    print(f"\n  {label} (degree={degree})")
    print(f"  {'Alpha':<12} {'CV MSE':>12} {'CV MSE Std':>12} {'CV R²':>10} {'CV R² Std':>10}")
    print("  " + "-" * 60)
    best_a, best_cv = None, -99
    for a in alphas:
        pipe = Pipeline([
            ("poly",   PolynomialFeatures(degree=degree, include_bias=True)),
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=a))
        ])
        cv_mse = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
        cv_r2  =  cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        print(
            f"  alpha={a:<9.0e}  "
            f"CV MSE={cv_mse.mean():8.4f}  ±{cv_mse.std():7.4f}  "
            f"CV R²={cv_r2.mean():.4f}  ±{cv_r2.std():.4f}"
        )
        if cv_r2.mean() > best_cv:
            best_cv, best_a = cv_r2.mean(), a
    print(f"\n  >>> Best alpha for {label}: {best_a}  (CV R² = {best_cv:.4f})")
    return best_a

print("=" * 70)
print("ALPHA (REGULARIZATION STRENGTH) TUNING — RIDGE REGRESSION")
print("=" * 70)
tune_alpha(X1, y1, degree=3, label="Var1: All 6 Features")
tune_alpha(X2, y2, degree=4, label="Var2: All 3 Coordinates")
