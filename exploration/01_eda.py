"""
Step 1: Exploratory Data Analysis (EDA)
========================================
First step taken after loading the datasets.
Checks:
  - Shape and column names
  - Missing/null values
  - Basic descriptive statistics
  - Feature value ranges
  - Linear Pearson correlations (feature vs target)

Run: python 01_eda.py
"""

import pandas as pd
import numpy as np

ROLL_NO = "BT2024105"
DATA_DIR = "../trainAndTestData"

# ─── Load all 4 files ────────────────────────────────────────────────────────
train1 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var1.csv")
test1  = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_test_var1.csv")
train2 = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_train_var2.csv")
test2  = pd.read_csv(f"{DATA_DIR}/{ROLL_NO}_test_var2.csv")

# ─── Shape & Columns ─────────────────────────────────────────────────────────
print("=" * 60)
print("DATASET SHAPES & COLUMNS")
print("=" * 60)
for name, df in [("train_var1", train1), ("test_var1", test1),
                  ("train_var2", train2), ("test_var2", test2)]:
    print(f"  {name}: {df.shape}  |  Columns: {df.columns.tolist()}")

# ─── Missing Values ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)
for name, df in [("train_var1", train1), ("train_var2", train2)]:
    nulls = df.isnull().sum().sum()
    print(f"  {name}: {nulls} missing values")

# ─── Descriptive Statistics ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VAR1 TRAIN — DESCRIPTIVE STATISTICS")
print("=" * 60)
print(train1.describe().round(4).to_string())

print("\n" + "=" * 60)
print("VAR2 TRAIN — DESCRIPTIVE STATISTICS")
print("=" * 60)
print(train2.describe().round(4).to_string())

# ─── Feature Range Check ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE RANGES (min / max per column)")
print("=" * 60)
print("\nvar1 train:")
for col in train1.columns:
    print(f"  {col}: [{train1[col].min():.4f}, {train1[col].max():.4f}]")

print("\nvar2 train:")
for col in train2.columns:
    print(f"  {col}: [{train2[col].min():.4f}, {train2[col].max():.4f}]")

# ─── Pearson Correlation with Target y ────────────────────────────────────────
print("\n" + "=" * 60)
print("LINEAR PEARSON CORRELATIONS WITH TARGET y")
print("=" * 60)
print("\nvar1 - Correlation of each feature with y:")
corr1 = train1.corr()["y"].drop("y")
for feat, val in corr1.items():
    print(f"  {feat}: {val:.6f}")

print("\nvar2 - Correlation of each feature with y:")
corr2 = train2.corr()["y"].drop("y")
for feat, val in corr2.items():
    print(f"  {feat}: {val:.6f}")

print("\n>>> OBSERVATION: All linear correlations are very weak.")
print(">>> Non-linear polynomial modeling is clearly needed.")
