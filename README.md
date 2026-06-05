# Insurance Charges Prediction — Linear Regression

> Predicting annual medical insurance charges using a production-grade sklearn pipeline with feature engineering, log-target transformation, and SelectKBest feature selection.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Approach & Methodology](#approach--methodology)
- [Problems Encountered & Solutions Applied](#problems-encountered--solutions-applied)
- [Feature Engineering](#feature-engineering)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)

---

## Overview

This project builds a supervised regression model to **predict individual medical insurance charges** from demographic and lifestyle features. The objective is to deliver a full, production-style ML pipeline — from raw CSV through EDA, preprocessing, feature engineering, model selection, and rigorous evaluation.

| Attribute | Detail |
|-----------|--------|
| **Task** | Regression (continuous target) |
| **Target** | `charges` (USD) |
| **Algorithm** | `LinearRegression` + `SelectKBest (f_regression)` |
| **Primary Metric** | R² Score (log scale) |
| **Secondary Metrics** | MAE ($), RMSE ($) |
| **Baseline** | Predict mean charges always |

---

## Dataset

**File:** `insurance.csv`  
**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)  
**Shape:** 1,338 rows × 7 columns

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Age of the primary beneficiary |
| `sex` | categorical | `male` / `female` |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents covered |
| `smoker` | categorical | `yes` / `no` |
| `region` | categorical | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | float | **Target** — Individual annual medical costs billed |

---

## Project Structure

```
insurance-charges-prediction/
│
├── Linear_regression_insurance_final.ipynb   # Full notebook
├── insurance.csv                              # Raw dataset
├── README.md                                  # This file
│
├── outputs/
│   ├── before_log_transform.png               # Skewness analysis plots
│   ├── skewness_check.png                     # Before/after log transform
│   └── final_diagnostics.png                  # 4-panel evaluation plots
```

---

## Approach & Methodology

The project follows a strict ML engineering workflow to prevent data leakage and ensure reliable evaluation metrics:

```
1. Problem Definition        →  Define task, target, metric, baseline
2. Data Loading & Audit      →  Shape, dtypes, nulls, duplicates, target stats
3. EDA                       →  Distribution plots, violin, pairplot, heatmap
4. Skewness Analysis         →  Quantify skew; decide on log transform
5. Outlier Detection         →  IQR screening + Z-score removal
6. Log Transform Target      →  y = log(charges) before any split
7. Duplicate Removal         →  df.drop_duplicates()
8. Feature Engineering       →  12 new features (interaction, polynomial, flags)
9. Train/Val/Test Split       →  80/10/10 BEFORE preprocessing (leakage prevention)
10. Preprocessing Pipeline   →  ColumnTransformer (impute + scale + encode)
11. Feature Selection        →  SelectKBest tuned via 5-fold CV
12. Model Training           →  LinearRegression.fit(X_train, log_y_train)
13. Evaluation               →  R², MAE($), RMSE($), CV, residual diagnostics
14. Diagnostic Plots         →  Actual vs Pred, Residuals, Histogram, CV bars
```

---

## Problems Encountered & Solutions Applied

###  Problem 1 — Severe Right-Skew in Target Variable

**Observation:** `charges` had a skewness of **+1.51**, driven by a small subset of very high-cost smoker/obese patients. Raw linear regression on this distribution violates the normality assumption for residuals.

**Solution:** Applied `log(charges)` as the training target. Skewness dropped from +1.51 to **~0.07** (near-normal). Final predictions are exponentiated back with `np.exp()` for dollar-scale reporting.

---

###  Problem 2 — Outliers in Numeric Features

**Observation:** IQR screening detected outliers in BMI and charges. These extreme values could disproportionately influence OLS coefficient estimates.

** Solution:** Z-score filtering (threshold = 3σ) applied to `age` and `bmi` columns only. `children` was excluded (skewness < 1.0, treated as categorical). `charges` outliers were handled by the log transform, not by row removal.

---

###  Problem 3 — Non-Linear Relationships

**Observation:** Pairplot with `hue='smoker'` revealed two completely separate charge distributions. The effect of BMI on charges multiplied dramatically for smokers — a pure linear model on raw features cannot capture this.

**Solution:** Engineered interaction terms (`smoker_bmi`, `smoker_age`, `smoker_bmi2`, `smoker_obese`) and polynomial terms (`age²`, `bmi²`) to explicitly model these non-linear effects inside a linear framework.

---

###  Problem 4 — Categorical Encoding

**Observation:** `sex`, `smoker`, `region`, and engineered categorical columns (`children_cat`, `age_group`, `risk_group`) could not be passed raw to a linear model. Label encoding would impose false ordinal relationships.

**Solution:** `OneHotEncoder(handle_unknown='ignore')` inside the sklearn pipeline's `ColumnTransformer`. `smoker` was also manually mapped to `smoker_yes` (0/1) for interaction term construction.

---

###  Problem 5 — Data Leakage Risk

**Observation:** A critical silent bug in many beginner pipelines — fitting `StandardScaler` or `SimpleImputer` on the full dataset before the train/test split leaks test-set statistics into the model, inflating evaluation metrics.

** Solution:** Train/test split is performed **before** any preprocessing. The `Pipeline` wrapping `ColumnTransformer` is `.fit()` only on `X_train`. Test and validation sets are `.transform()`-only.

---

###  Problem 6 — Feature Redundancy

**Observation:** After feature engineering and one-hot encoding, the expanded feature matrix had 20+ columns. Including all of them adds noise and can inflate variance of OLS coefficients.

** Solution:** `SelectKBest(f_regression)` added as a pipeline stage. Best K was determined by looping over K ∈ {8, 10, 12, 15, 18, 20} using 5-fold cross-validation on the training set, selecting the K with the highest mean R².

---

###  Problem 7 — Risk Cluster Separation

**Observation:** EDA clearly showed three distinct charge clusters: non-smokers (low), smokers with BMI < 30 (medium), smokers with BMI ≥ 30 (high). Without explicit cluster features, the model treats all patients on a uniform continuum.

** Solution:** Engineered `risk_group` feature with three levels (`low`, `medium`, `high`) based on smoker status and obesity threshold. This provides the model with direct cluster-level information, improving performance across all three charge bands.

---

## Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `smoker_yes` | `smoker.map({'yes':1,'no':0})` | Integer for interaction term construction |
| `age2` | `age²` | Accelerating cost increase with age |
| `bmi2` | `bmi²` | Non-linear BMI-cost relationship |
| `obese` | `bmi ≥ 30` → 1/0 | Industry underwriting threshold |
| `smoker_bmi` | `smoker_yes × bmi` | Dominant driver of high charges |
| `smoker_age` | `smoker_yes × age` | Older smokers face compounding risk |
| `smoker_bmi2` | `smoker_yes × bmi²` | Smoker-BMI curvature |
| `smoker_obese` | `smoker_yes × obese` | Hard interaction at BMI=30 threshold |
| `age_bmi` | `age × bmi` | Joint aging + weight effect |
| `children_cat` | `children.astype(str)` | Treats children as nominal (skew < 1) |
| `age_group` | bins: young/adult/old | Coarse age segmentation |
| `risk_group` | low / medium / high | Explicit 3-cluster label |

---

## Model Pipeline

```python
model = Pipeline([
    ("preprocessing", ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])),
    ("selector",   SelectKBest(f_regression, k=best_k)),
    ("regressor",  LinearRegression())
])
```

**Data Split:**
```
Total: 1,338 rows
Train : 80%  (~1,070 rows) — fit pipeline + model
Val   : 10%  (~134 rows)  — tune best_k
Test  : 10%  (~134 rows)  — final held-out evaluation
```

---

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Train R² (log) | ≥ 0.86 | Strong fit on training data |
| Test R² (log) | ≥ 0.85 | Strong generalization |
| Train–Test Gap | < 0.05 |  No overfitting |
| 5-Fold CV Mean R² | ≥ 0.85 |  Stable across folds |
| CV Std R² | < 0.02 |  Low variance |
| MAE (dollars) | ~$2,500–$3,000 | Average prediction error |
| Residual Mean | ~0.00 |  Unbiased predictions |
| Residual Skewness | ~0.00 |  Normal residuals confirmed |

**Key Diagnostic Checks Passed:**
-  Residual plot shows random scatter (homoscedasticity)
-  Residual histogram is bell-shaped (normality)
-  Actual vs Predicted ($) points tight around 45° line
-  CV bar chart shows consistent R² across all 5 folds

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/insurance-charges-prediction.git
cd insurance-charges-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn missingno
```

### 4. Add the dataset
Place `insurance.csv` in the project root directory.  
Download from: https://www.kaggle.com/datasets/mirichoi0218/insurance

### 5. Run the notebook
```bash
jupyter notebook Linear_regression_insurance_final.ipynb
```

Run all cells top-to-bottom. Output plots are saved as `.png` files automatically.

---

## Key Learnings

1. **Log-transform skewed targets first.** It is the single highest-impact preprocessing step for insurance and financial prediction tasks.
2. **Build interaction terms manually when domain knowledge justifies it.** `smoker × bmi` outperformed all raw features combined.
3. **Split before you preprocess.** Fitting scalers/imputers on the full dataset is a silent bug that inflates metrics.
4. **SelectKBest is a lightweight, effective filter.** Tuning K via CV is more principled than manually choosing columns.
5. **Explicit cluster features help linear models.** `risk_group` gave the model what a decision tree would learn implicitly.

---


*Built with Python · scikit-learn · pandas · seaborn · matplotlib*
