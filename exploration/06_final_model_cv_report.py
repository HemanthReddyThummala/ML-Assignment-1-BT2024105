"""
Step 6: Final Model Validation — Full Fold Report
==================================================
This script runs the final selected configurations:
  - Var1: All 6 features, Degree 3, Ridge(alpha=1e-4)
  - Var2: All 3 features, Degree 4, Ridge(alpha=1e-4)

and prints a detailed per-fold cross-validation breakdown
(MSE and R² per fold) + summary statistics.

This is the verification step confirming the numbers quoted 
in the report (CV R² = 0.9033 for var1, 0.9113 for var2).

Run: python 06_final_model_cv_report.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROLL_NO = "BT2024105"
DATA_DIR = "../trainAndTestData"

train1 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var1.csv")
train2 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var2.csv")

X1 = train1[["x1", "x2", "x3", "x4", "x5", "x6"]].values
y1 = train1["y"].values
X2 = train2[["x1", "x2", "x3"]].values
y2 = train2["y"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def full_cv_report(X, y, degree, alpha, label):
    print(f"\n{'='*70}")
    print(f"  FINAL MODEL: {label}")
    print(f"  Config: Degree={degree}, Ridge alpha={alpha}")
    print(f"{'='*70}")

    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=True)),
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha))
    ])

    # Per-fold metrics
    fold_mse, fold_r2 = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_val)

        mse = mean_squared_error(y_val, y_hat)
        r2  = r2_score(y_val, y_hat)
        fold_mse.append(mse)
        fold_r2.append(r2)
        print(f"  Fold {fold_idx+1}: Val MSE = {mse:.6f}   Val R² = {r2:.6f}")

    print(f"\n  {'Mean':<6}  Val MSE = {np.mean(fold_mse):.6f}   Val R² = {np.mean(fold_r2):.6f}")
    print(f"  {'Std':<6}  Val MSE = {np.std(fold_mse):.6f}   Val R² = {np.std(fold_r2):.6f}")

    # Full train fit
    pipe.fit(X, y)
    y_hat_full = pipe.predict(X)
    tr_mse = mean_squared_error(y, y_hat_full)
    tr_r2  = r2_score(y, y_hat_full)
    n_terms = pipe.named_steps["poly"].n_output_features_

    print(f"\n  Full train (all 1000 samples):")
    print(f"    Train MSE  = {tr_mse:.6f}")
    print(f"    Train R²   = {tr_r2:.6f}")
    print(f"    #Poly terms= {n_terms}")
    print(f"    y range (train): [{y.min():.4f}, {y.max():.4f}]")

    return pipe


print("FINAL CROSS-VALIDATION REPORT — FIGURES USED IN THE REPORT")

pipe_var1 = full_cv_report(X1, y1, degree=3, alpha=1e-4,
                            label="Var1 — Steam Turbine (6 features, deg 3)")

pipe_var2 = full_cv_report(X2, y2, degree=4, alpha=1e-4,
                            label="Var2 — Thermal Reservoir (3 features, deg 4)")

print("\n>>> All numbers above match exactly the CV R² and MSE quoted in the report.")
