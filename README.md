# CS301 Machine Learning — Assignment 1: Polynomial Regression

**Student Roll Number:** `BT2024105`  
**Semester:** 5  
**GitHub Repository:** [https://github.com/HemanthReddyThummala/ML-Assignment-1-BT2024105](https://github.com/HemanthReddyThummala/ML-Assignment-1-BT2024105)  

---

## 1. Project Summary & Selected Models

| Problem ID | Task Description | Selected Features | Polynomial Degree | Model Pipeline | 5-Fold CV R² | 5-Fold CV MSE |
|---|---|---|---|---|---|---|
| **var1** | Steam Turbine Optimization (Net Power Score) | All 6 features (`x1`–`x6`) | **Degree 3** | `Poly(3) + Scaler + Ridge(1e-4)` | **0.9033 ± 0.009** | **1.0083 ± 0.077** |
| **var2** | Subterranean Thermal Mapping (Anomaly Score) | All 3 coordinates (`x1`, `x2`, `x3`) | **Degree 4** | `Poly(4) + Scaler + Ridge(1e-4)` | **0.9113 ± 0.011** | **3.5680 ± 0.240** |

---

## 2. Evaluation of Assignment Hints (Ablation Analysis)

The assignment prompt provided two guidance hints:
- **Phase 1 (var1):** *"Optimal results using a polynomial of degree 3 and first 3 of the features given"*
  - **Empirical Test:**
    - Model A (first 3 features `x1, x2, x3`, Degree 3): 5-Fold CV R² = **0.2062**, CV MSE = **8.3411** *(Severe underfitting, explains only ~20% of variance)*.
    - Model B (all 6 features `x1`–`x6`, Degree 3): 5-Fold CV R² = **0.9033**, CV MSE = **1.0083** *(Selected: over 90% variance captured)*.

- **Phase 2 (var2):** *"Optimal results using a polynomial of degree 4 and only the first feature given"*
  - **Empirical Test:**
    - Model A (single coordinate `x1` only, Degree 4): 5-Fold CV R² = **0.0487**, CV MSE = **38.5357** *(Severe underfitting, explains <5% of variance)*.
    - Model B (all 3 coordinates `x1, x2, x3`, Degree 4): 5-Fold CV R² = **0.9113**, CV MSE = **3.5680** *(Selected: over 91% variance captured with 35 terms)*.

---

## 3. Environment Setup & Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`.

---

## 4. How to Run Training & Generate Predictions

To train both optimal pipelines and generate the final prediction CSV files, execute:

```bash
python test_and_predict.py
```

This outputs:
- `BT2024105_pred_var1.csv`
- `BT2024105_pred_var2.csv`

Both files are validated against `sample_submission.csv` (single header `y`, exactly 1,000 prediction rows, no missing values).

---

## 5. Intermediate Exploration & Ablation Scripts

To verify every empirical claim, metric, and degree ablation cited in the report, run the scripts in `exploration/`:

```bash
cd exploration

# Step 1: Exploratory Data Analysis (shapes, missing values, descriptive statistics)
python 01_eda.py

# Step 2: Var 1 feature subset vs. all features ablation across degrees 1-5
python 02_var1_feature_degree_ablation.py

# Step 3: Var 2 single coordinate vs. 3D coordinates ablation across degrees 1-6
python 03_var2_feature_degree_ablation.py

# Step 4: Ridge regularization parameter alpha tuning (1e-6 to 100)
python 04_alpha_tuning.py

# Step 5: Regularization comparison (Ridge vs. Lasso vs. ElasticNet)
python 05_ridge_vs_lasso_elasticnet.py

# Step 6: Full 5-fold cross-validation report with per-fold metrics breakdown
python 06_final_model_cv_report.py
```

---

## 6. Deliverables & Directory Structure

```
.
├── test_and_predict.py           # End-to-end training and inference pipeline
├── requirements.txt              # Project dependencies (numpy, pandas, scikit-learn, matplotlib)
├── README.md                     # Project documentation
├── BT2024105_Report.pdf          # 4-page academic report with embedded matplotlib figures
├── BT2024105_pred_var1.csv       # Optimal test predictions for var1 (Steam Turbine)
├── BT2024105_pred_var2.csv       # Optimal test predictions for var2 (Thermal Reservoir)
├── figures/                      # High-resolution (300 DPI) publication plots
├── exploration/                  # 6 intermediate empirical analysis scripts
└── trainAndTestData/             # Assigned train and test CSV datasets
```

---

## 7. Submission Checklist

- [x] **Prediction File 1:** `BT2024105_pred_var1.csv` (1,000 predictions, matches sample submission format)
- [x] **Prediction File 2:** `BT2024105_pred_var2.csv` (1,000 predictions, matches sample submission format)
- [x] **Report PDF:** `BT2024105_Report.pdf` (4-page concise academic write-up with embedded matplotlib figures and rationale)
- [x] **GitHub Repository:** [https://github.com/HemanthReddyThummala/ML-Assignment-1-BT2024105](https://github.com/HemanthReddyThummala/ML-Assignment-1-BT2024105) (all training, inference, and exploration code)

