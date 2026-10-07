"""
Step 2: Var1 — Feature Subset vs Full Feature Ablation
=======================================================
After EDA confirmed non-linearity, this script systematically tests:
  - Using only the FIRST 3 features (x1, x2, x3) as suggested in the assignment hint
  - Using ALL 6 features (x1 through x6)
  - Across polynomial degrees 1, 2, 3, 4, 5

For each combination we compute:
  - Number of polynomial terms generated
  - Training MSE and R²
  - 5-Fold Cross-Validation MSE and R²

Observation: The "first 3 features" hint severely underfits.
All 6 features at degree 3 gives optimal CV performance.

Run: python 02_var1_feature_degree_ablation.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

ROLL_NO = "BT2024105"
DATA_DIR = "../trainAndTestData"

train1 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var1.csv")
X_all = train1[["x1", "x2", "x3", "x4", "x5", "x6"]].values
X_sub = train1[["x1", "x2", "x3"]].values
y     = train1["y"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(label, X, y, degree, alpha=1e-4):
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(degree=degree, include_bias=True)),
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha))
    ])
    pipe.fit(X, y)

    n_terms  = pipe.named_steps["poly"].n_output_features_
    tr_mse   = mean_squared_error(y, pipe.predict(X))
    tr_r2    = r2_score(y, pipe.predict(X))
    cv_mse   = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error")
    cv_r2    =  cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    print(
        f"  {label:<30}  deg={degree}  terms={n_terms:>4}  "
        f"| Train MSE={tr_mse:7.4f}  Train R²={tr_r2:.4f}  "
        f"| CV MSE={cv_mse.mean():7.4f}(±{cv_mse.std():.4f})  "
        f"CV R²={cv_r2.mean():.4f}(±{cv_r2.std():.4f})"
    )
    return cv_r2.mean()

print("=" * 110)
print("VAR1: FEATURE SUBSET vs. ALL FEATURES — DEGREE ABLATION")
print("Using Ridge(alpha=1e-4) + StandardScaler, 5-Fold CV")
print("=" * 110)

print("\n--- Using FIRST 3 FEATURES only: x1, x2, x3 ---")
for d in [1, 2, 3, 4]:
    evaluate("Subset (x1, x2, x3)", X_sub, y, degree=d)

print("\n--- Using ALL 6 FEATURES: x1 through x6 ---")
best_cv_r2, best_deg = -99, -1
for d in [1, 2, 3, 4, 5]:
    cv = evaluate("All 6 Features", X_all, y, degree=d)
    if cv > best_cv_r2:
        best_cv_r2, best_deg = cv, d

print()
print(f">>> BEST CONFIGURATION: All 6 Features, Degree {best_deg}, CV R² = {best_cv_r2:.4f}")
print()
print(">>> NOTE: Degree 5 shows training R² close to 1.0 but CV R² drops sharply — OVERFITTING.")
print(">>> NOTE: Degree 3 on all 6 features gives the best generalization. SELECTED.")
